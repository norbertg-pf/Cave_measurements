-- Shared variable for sending data
local share_send_now = 0

function main()

    reset()

    channel.open("allslots")

    local v_threshold = 0.02 --TRESHOLD
    -- Number of measurements in graph buffer
    local N = 10000

    -- desired sampling interval (seconds)
    local dt = SAMPLE_RATE          -- 100 ms

    -- overhead for instrument processing
    local overhead = 0.003   -- 3 ms margin

    -- read instrument line frequency (50 or 60 Hz)
    local line_freq = 50

    -- Create reading buffer
    QDbuf = buffer.make(N, buffer.STYLE_WRITABLE_FULL)
    buffer.write.format(QDbuf, buffer.UNIT_VOLT, buffer.DIGITS_4_5)
    QDbuf.clear()
    QDbuf.fillmode = buffer.FILL_CONTINUOUS

    ----------------------------------------------------------------------
    -- Display setup
    ----------------------------------------------------------------------
    display.changescreen(display.SCREEN_GRAPH)
    display.activebuffer = QDbuf

    ----------------------------------------------------------------------
    -- Create scan buffer
    ----------------------------------------------------------------------
    defbuffer1.clear()
    defbuffer1.capacity = 1000

    ----------------------------------------------------------------------
    -- TIMER TRIGGER SETUP
    ----------------------------------------------------------------------
    trigger.clear()

    -- Use Timer as trigger source
    trigger.timer[1].delay = dt       -- 100 ms period
    trigger.timer[1].count = 0        -- infinite
    trigger.timer[1].start.stimulus = trigger.EVENT_TIMER1

    -- Start the timer
    trigger.timer[1].enable = trigger.ON


    timer.cleartime()

    ---------------------------------------------------------------------
    -- COMPUTE MAXIMUM ALLOWABLE NPLC BASED ON dt
    ---------------------------------------------------------------------
    local function compute_nplc(dt, line_freq, overhead)
        -- maximum NPLC that fits within dt
        local max_nplc = (dt - overhead) * line_freq
        
        -- enforce lower bound (minimum valid NPLC 0.01)
        if max_nplc < 0.01 then
            max_nplc = 0.01
        end

        -- clamp to realistic upper range for DAQ6510 (0.01 to 10)
        if max_nplc > 10 then
            max_nplc = 10
        end

        return max_nplc
    end

    local function adjust_dt(t)
        local elapsed = timer.gettime() - t
        local adj_t = dt + dt - elapsed
        if adj_t < 0.0001 then    -- realistic minimum delay (1 ms)
            return 0.0001
        else
            return adj_t
        end
    end

    ----------------------------------------------------------------------
    -- Configure DMM
    ----------------------------------------------------------------------
    local auto_nplc = compute_nplc(dt, line_freq, overhead)
    dmm.measure.func = dmm.FUNC_DC_VOLTAGE
    dmm.measure.nplc = auto_nplc

    ----------------------------------------------------------------------
    -- MAIN LOOP
    ----------------------------------------------------------------------
    while true do
        local t = timer.gettime()
        -- Wait for the next timer event
        -- trigger.timer[1].wait(1)

        -- Take measurement
        local v = dmm.measure.read()

        -- Store in buffer
        buffer.write.reading(QDbuf, v, t)

        -- Prepare for LAN sender thread
        share_data = string.format("%.9f", v)
        print(share_data)
        -- trigger.timer[1].delay = adjust_dt(t)        -- 100 ms period
        delay(adjust_dt(t))

    end
end

main()
