-- version 1.3
-- 2024-05-08
-- xiaozheng.zhang
-- This script applies to falconi/g/k falcon2.1 robinw
-- update the struct of InnoCoChannelPoint in robinw_generic

-- version 1.2
-- 2023-11-03
-- Feng Yang
-- This script applies to falconi/g/k falcon2.1 robinw
-- update the struct of InnoEnChannelPoint and InnoEnXyzPoint in falcon2.1 and robinw

-- You can register the protocol to a TCP port or UDP port by changing the tcpopen variable.
tcpopen = false

do

    --[[ ---------------------------------------------------------------------
            /* 17 bytes per block header */
            struct InnoBlockHeader
            {
                // unit is PI/32768
                int16_t h_angle;
                int16_t v_angle;

                uint16_t ts_10us;
                uint16_t scan_idx;

                uint16_t scan_id: 9;

                // real angle is h_angle + h_angle_diff_1
                int64_t h_angle_diff_1: 9;
                int64_t h_angle_diff_2: 10;
                int64_t h_angle_diff_3: 11;

                // real angle is v_angle + v_angle_diff_1 + kVAngleDiffBase * channel
                int64_t v_angle_diff_1: 8;  // 196 + [-128, 127]
                int64_t v_angle_diff_2: 9;  // 392 + [-256, 255]
                int64_t v_angle_diff_3: 9;  // 588 + [-256, 255]

                uint64_t in_roi: 2;
                uint64_t facet: 3;
                uint64_t reserved_flags: 2;
            };


			--falcon2.1 and robinw	18 bytes per EnBlock header
			typedef DEFINE_INNO_COMPACT_STRUCT(InnoEnBlockHeader) {
				  /* horizontal angle, 0 is straight forward, right is positive,
					 unit is kRadPerInnoAngleUnit rad, range (-PI to PI) */
				  int16_t h_angle;
				  /* vertical angle, 0 is the horizon, up is positive,
					 unit is kRadPerInnoAngleUnit rad, range (-PI to PI) */
				  int16_t v_angle;
				  /* relative timestamp (to ts_start_us) in 10us, 0-655,350us */
				  uint16_t ts_10us;
				  // real angle is h_angle + h_angle_diff_1
				  int64_t h_angle_diff_1 : 11;
				  int64_t h_angle_diff_2 : 11;
				  int64_t h_angle_diff_3 : 12;
				  // real angle is v_angle + v_angle_diff_1 + kVAngleDiffBase * channel
				  int64_t v_angle_diff_1 : 10;
				  int64_t v_angle_diff_2 : 10;
				  int64_t v_angle_diff_3 : 10;
				  uint16_t scan_idx;            /* point idx within the scan line */
				  uint16_t scan_id : 9;         /* id of the scan line */
				  /*   0: in sparse region
				  0x01: in vertical slow region
				  0x11: in center ROI
				  only for falcon  */
				  uint16_t in_roi : 2;
				  uint16_t facet : 3;
				  uint16_t reserved_flags : 2; /* all 0 */
                  }


            /* 10 bytes per CoBlock header */
            typedef DEFINE_INNO_COMPACT_STRUCT(InnoCoBlockHeader) {
            /* polygon angle, 0 is straight forward,
                unit is kRadPerInnoAngleUnit rad, range (-PI to PI) */
            int16_t p_angle;
            int16_t g_angle;
            /* relative timestamp (to ts_start_us) in 10us, 0-655,350us */
            uint16_t ts_10us;
            uint16_t scan_idx;    /* point idx within the scan line */
            uint16_t scan_id : 9; /* id of the scan line */
            /*   0: in sparse region
            0x01: in vertical slow region
            0x11: in center ROI
            only for falcon  */
            uint16_t in_roi : 2;
            uint16_t facet : 3;
            uint16_t reserved_flags : 2; /* all 0 */
            }
            InnoCoBlockHeader;
    --]] ---------------------------------------------------------------------

    local field_bh_h_angle = ProtoField.int16("Innovusion", "h_angle", base.DEC, nil, nil, "horizontal angle, 0 is straight forward, right is positive, unit is [PI/32768] rad, range (-PI to PI)")
    local field_bh_v_angle = ProtoField.int16("Innovusion", "v_angle", base.DEC, nil, nil, "vertical angle, 0 is the horizon, up is positive, unit is [PI/32768] rad, range (-PI to PI)")

    local field_bh_ts_10us = ProtoField.uint16("Innovusion", "ts_10us", base.DEC, nil, nil, "relative timestamp (to ts_start_us) in 10us, 0-655350us")
	local type_bh_in_roi = {
         [0] = "in sparse region",
         [1] = "in vertical slow region",
         [2] = "in horizontal slow region",
         [3] = "in center ROI"
        }
	local field_bh_scan_idx = ProtoField.uint16("Innovusion", "scan_idx", base.DEC, nil, nil, "point idx within the scan line")



	-- 8
    local field_bh_scan_id = ProtoField.uint16("Innovusion", "scan_id", base.DEC, nil, 0x01FF, "id of the scan line")
	    field_bh_h_angle_diff_1 = ProtoField.int32("Innovusion", "h_angle_diff_1", base.DEC, nil, 0x000003FE, "h_angle + h_angle_diff_1")
        field_bh_h_angle_diff_2 = ProtoField.int32("Innovusion", "h_angle_diff_2", base.DEC, nil, 0x000FFC00, "h_angle + h_angle_diff_2")

    -- 11
        field_bh_h_angle_diff_3 = ProtoField.int32("Innovusion", "h_angle_diff_3", base.DEC, nil, 0x7FF00000, "h_angle + h_angle_diff_3")
        field_bh_v_angle_diff_1 = ProtoField.int32("Innovusion", "v_angle_diff_1", base.DEC, nil, 0x00007F80, "196 + [-128, 127]")
        field_bh_v_angle_diff_2 = ProtoField.int32("Innovusion", "v_angle_diff_2", base.DEC, nil, 0x00FF8000, "392 + [-256, 255]")

    -- 15
        field_bh_v_angle_diff_3 = ProtoField.int16("Innovusion", "v_angle_diff_3", base.DEC, nil, 0x01FF, "588 + [-256, 255]")

        local field_bh_in_roi = ProtoField.uint8("Innovusion", "in_roi", base.DEC, type_bh_in_roi, 0x06, "")
        local field_bh_facet = ProtoField.uint8("Innovusion", "facet", base.DEC, nil,  0x38, "")
        local field_bh_reserved_flags = ProtoField.uint8("Innovusion", "reserved_flags", base.DEC, nil, 0xC0, "all 0")

		field_bh_h_angle_diff_11 = ProtoField.int32("Innovusion", "h_angle_diff_1", base.DEC, nil, 0x000007FF, "h_angle + h_angle_diff_1")
        field_bh_h_angle_diff_21 = ProtoField.int32("Innovusion", "h_angle_diff_2", base.DEC, nil, 0x003FF800, "h_angle + h_angle_diff_2")
	--11 11

        field_bh_h_angle_diff_31 = ProtoField.int32("Innovusion", "h_angle_diff_3", base.DEC, nil, 0x0003FFC0, "h_angle + h_angle_diff_3")
        field_bh_v_angle_diff_11 = ProtoField.int32("Innovusion", "v_angle_diff_1", base.DEC, nil, 0x0FFC0000, "196 + [-128, 127]")
	--12 10

        field_bh_v_angle_diff_21 = ProtoField.int32("Innovusion", "v_angle_diff_2", base.DEC, nil, 0x00003FF0, "392 + [-256, 255]")
        field_bh_v_angle_diff_31 = ProtoField.int16("Innovusion", "v_angle_diff_3", base.DEC, nil, 0x00FFC000, "588 + [-256, 255]")


    --rw_generic
    field_bh_p_angle_diff_rw_generic = ProtoField.int16("Innovusion", "p_angle", base.DEC, nil, nil, "")
    field_bh_g_angle_diff_rw_generic = ProtoField.int16("Innovusion", "g_angle", base.DEC, nil, nil, "")

	--end
	--
    -- size : 17(falconigk) 18(robinw falcon2.1)
    --


    function InnoBlockHeader_dissector(buf,pkt,parent)
        if is_falcon_igk == true then
			m = parent:add("InnoBlockHeader", buf(0,17))
        elseif is_robinw_generic == true then
            m = parent:add("InnoCoBlockHeader", buf(0,10))
		else
			m = parent:add("InnoBlockHeader", buf(0,18))
		end



		if is_falcon_igk == true
		then
            m:add_le(field_bh_h_angle, buf(0,2))
            m:add_le(field_bh_v_angle, buf(2,2))
            m:add_le(field_bh_ts_10us, buf(4,2))
			m:add_le(field_bh_scan_idx, buf(6,2))

			-- 8
            m:add_le(field_bh_scan_id, buf(8,2))
			m:add_le(field_bh_h_angle_diff_1, buf(9,4))
			m:add_le(field_bh_h_angle_diff_2, buf(9,4))

			-- 11
			m:add_le(field_bh_h_angle_diff_3, buf(9,4))
			m:add_le(field_bh_v_angle_diff_1, buf(12,4))
			m:add_le(field_bh_v_angle_diff_2, buf(12,4))

			-- 15
			m:add_le(field_bh_v_angle_diff_3, buf(15,2))
			m:add_le(field_bh_in_roi, buf(16,1))
			m:add_le(field_bh_facet, buf(16,1))
			m:add_le(field_bh_reserved_flags, buf(16,1))
        elseif is_robinw_generic == true then
            m:add_le(field_bh_p_angle_diff_rw_generic, buf(0,2))
            m:add_le(field_bh_g_angle_diff_rw_generic, buf(2,2))
            m:add_le(field_bh_ts_10us, buf(4,2))
            m:add_le(field_bh_scan_idx, buf(6,2))
            m:add_le(field_bh_scan_id, buf(8,2))
            m:add_le(field_bh_in_roi, buf(8,2))
            m:add_le(field_bh_facet, buf(8,2))
            m:add_le(field_bh_reserved_flags, buf(8,2))
		else
            m:add_le(field_bh_h_angle, buf(0,2))
            m:add_le(field_bh_v_angle, buf(2,2))
            m:add_le(field_bh_ts_10us, buf(4,2))
			m:add_le(field_bh_h_angle_diff_11, buf(6,4))
			m:add_le(field_bh_h_angle_diff_21, buf(6,4))


			m:add_le(field_bh_h_angle_diff_31, buf(8,4))
			m:add_le(field_bh_v_angle_diff_11, buf(8,4))


			m:add_le(field_bh_v_angle_diff_21, buf(10,4))
			m:add_le(field_bh_v_angle_diff_31, buf(10,4))

			m:add_le(field_bh_scan_idx, buf(14,2))

			m:add_le(field_bh_scan_id, buf(16,2))
			m:add_le(field_bh_in_roi, buf(16,2))
			m:add_le(field_bh_facet, buf(16,2))
			m:add_le(field_bh_reserved_flags, buf(16,2))
		end
    end


    --[[ ---------------------------------------------------------------------
            /* compact format, 4 bytes per point */
            struct InnoChannelPoint
            {
                uint32_t radius: 17;
                uint32_t refl: 8;
                uint32_t is_2nd_return: 1;
                uint32_t type: 2;
                uint32_t elongation: 4;
            };

			/* compact format, 8 bytes per point */
				typedef DEFINE_INNO_COMPACT_STRUCT(InnoEnChannelPoint) {
				  uint16_t reflectance;            /* reflectance, falcon 1-65535,robin 1-4095  */
				  uint16_t intensity;           /* intensity, falcon 1-65535,robin 1-4095  */
				  uint32_t elongation: 7;      /* elongation unit: 1ns */
				  uint32_t is_2nd_return: 1;    /* 0: 1st return, 1: 2nd return                  */
				  uint32_t radius : 19;         /* distance in distance unit, distance unit:1/400m, range [0, 655.35m] */
				  uint32_t type : 2;            /* 0: normal, 1: ground, 2: fog                  */
				  uint32_t firing: 1;           /* 0: weak, 1: strong */
				  uint32_t reserved_flags : 2;  /* all 0 */
				} InnoEnChannelPoint;

            /* compact format, 4 bytes per point */
            typedef DEFINE_INNO_COMPACT_STRUCT(InnoCoChannelPoint) {
            uint32_t refl : 12;         /* reflectance or intensity robin 1-4095  */
            uint32_t radius : 18;       /* distance in distance unit, distance unit:1/400m, range [0, 655.35m] */
            uint32_t is_2nd_return : 1; /* 0: 1st return, 1: 2nd return                  */
            uint32_t firing : 1;        /* 0: weak, 1: strong */
            }
            InnoCoChannelPoint;
    --]] ---------------------------------------------------------------------
	local type_channel_is_2nd_return = {
			[0] = "1st return",
			[1] = "2nd return",
		}
	local type_channel_type = {
			[0] = "normal",
			[1] = "ground",
			[2] = "fog"
		}
	local type_firing_type = {
			[0] = "weak",
			[1] = "strong",
	}

		field_channel_radius = ProtoField.uint32("Innovusion", "radius", base.DEC, nil, 0x0001FFFF, "distance in distance unit, range [0, 655.35m]")
		field_channel_refl = ProtoField.uint32("Innovusion", "refl", base.DEC, nil, 0x01FE0000, "reflectance, 1-254, 255 means a reflector or intensity, also 1-254 & 255=reflector")


		field_channel_is_2nd_return = ProtoField.uint32("Innovusion", "is_2nd_return", base.DEC, type_channel_is_2nd_return, 0x02000000, "0: 1st return, 1: 2nd return")


		field_channel_type = ProtoField.uint32("Innovusion", "type", base.DEC, type_channel_type, 0x0C000000, "0: normal, 1: ground, 2: fog")
		field_channel_elongation = ProtoField.uint32("Innovusion", "elongation", base.DEC, nil, 0xF0000000, "")

		field_channel_refl1 = ProtoField.uint32("Innovusion", "refl", base.DEC, nil, nil, "reflectance, 1-65535")
		field_channel_intensity = ProtoField.uint32("Innovusion", "intensity", base.DEC, nil, nil, "intensity, 1-65535")

		field_channel_elongation1 = ProtoField.uint32("Innovusion", "elongation", base.DEC, nil, 0x7F, "")
		field_channel_is_2nd_return1 = ProtoField.uint32("Innovusion", "is_2nd_return", base.DEC, type_channel_is_2nd_return, 0x80, "0: 1st return, 1: 2nd return")

		field_channel_radius1 = ProtoField.uint32("Innovusion", "radius", base.DEC, nil, 0x7FFFF, "distance in distance unit, range [0, 655.35m]")
		field_channel_type1 = ProtoField.uint32("Innovusion", "type", base.DEC, type_channel_type, 0x180000, "0: normal, 1: ground, 2: fog")
		field_channel_firing = ProtoField.uint32("Innovusion", "firing", base.DEC, type_firing_type, 0x200000, "0:weak, 1:strong")
		field_channel_reserved = ProtoField.uint32("Innovusion", "reserced", base.DEC, nil, 0xC00000, "all 0")

        field_channel_refl_rw_generic = ProtoField.uint32("Innovusion", "refl", base.DEC, nil, 0x00000FFF, "reflectance, 1-4095")
        field_channel_radius_rw_generic = ProtoField.uint32("Innovusion", "radius", base.DEC, nil, 0x3FFFF000, "distance in distance unit, range [0, 655.35m]")
        field_channel_is_2nd_return_rw_generic = ProtoField.uint32("Innovusion", "is_2nd_return", base.DEC, type_channel_is_2nd_return, 0x40000000, "0: 1st return, 1: 2nd return")
        field_channel_firing_rw_generic = ProtoField.uint32("Innovusion", "type", base.DEC, type_channel_type, 0x80000000, "0: weak, 1: strong")
    --
    -- size : 4(falconigk rw-generic) 8(robinw falcon2.1)
    --
    function InnoChannelPoint_dissector(buf,pkt,parent, index)
		if is_falcon_igk == true
		then
		    local m = parent:add("InnoChannelPoint[" .. tostring(index) .. "]")
			m:add_le(field_channel_radius,buf(0,4))
			m:add_le(field_channel_refl,buf(0,4))
			m:add_le(field_channel_is_2nd_return,buf(0,4))
			m:add_le(field_channel_type,buf(0,4))
			m:add_le(field_channel_elongation,buf(0,4))
		elseif is_robinw_generic == true then
		    local m = parent:add("InnoCoChannelPoint[" .. tostring(index) .. "]")
            m:add_le(field_channel_radius_rw_generic, buf(0,4))
            m:add_le(field_channel_refl_rw_generic, buf(0,4))
            m:add_le(field_channel_is_2nd_return_rw_generic, buf(0,4))
            m:add_le(field_channel_firing_rw_generic, buf(0,4))
        else
			local m = parent:add("InnoEnChannelPoint[" .. tostring(index) .. "]")
			m:add_le(field_channel_refl1, buf(0,2))
			m:add_le(field_channel_intensity, buf(2,2))

			m:add_le(field_channel_elongation1, buf(4,1))
			m:add_le(field_channel_is_2nd_return1, buf(4,1))

			m:add_le(field_channel_radius1, buf(5,3))
			m:add_le(field_channel_type1, buf(5,3))
			m:add_le(field_channel_firing, buf(5,3))
			m:add_le(field_channel_reserved, buf(5,3))
		end
    end


    --[[ ---------------------------------------------------------------------
            struct InnoBlock1
            {
                InnoBlockHeader header;
                InnoChannelPoint points[4];
            };

            struct InnoBlock2
            {
                InnoBlockHeader header;
                InnoChannelPoint points[8];
            };

            /* 10 + 4 * 8 = 42 bytes, 5.25 bytes/point */
            typedef DEFINE_INNO_COMPACT_STRUCT(InnoCoBlock1) {
                InnoCoBlockHeader header;
                InnoCoChannelPoint points[8];
            }
            InnoCoBlock1;

            /* 10 + 4 * 16 = 74 bytes */
            typedef DEFINE_INNO_COMPACT_STRUCT(InnoCoBlock2) {
                InnoCoBlockHeader header;
                InnoCoChannelPoint points[16];
            }
            InnoCoBlock2;
    --]] ---------------------------------------------------------------------

    -- 33 = 17 + 4 * 4  falconigk
	-- 18+4*8=50  robinw falcon2.1
    function InnoBlock1_dissector(buf, pkt, parent, index)

		if is_falcon_igk == true
		then
			local m = parent:add("InnoBlock1[" .. tostring(index) .. "]")
			InnoBlockHeader_dissector(buf(0,17),pkt,m)
			for i=0,3,1 do
				InnoChannelPoint_dissector(buf(17 + i * 4, 4), pkt, m, i)
			end
        elseif is_robinw_generic == true then
			local m = parent:add("InnoCoBlock1[" .. tostring(index) .. "]")
            InnoBlockHeader_dissector(buf(0,10),pkt,m)
            for i=0,7,1 do
                InnoChannelPoint_dissector(buf(10 + i * 4, 4), pkt, m, i)
            end
		else
			local m = parent:add("InnoEnBlock1[" .. tostring(index) .. "]")
			InnoBlockHeader_dissector(buf(0,18),pkt,m)
			for i=0,3,1 do
				InnoChannelPoint_dissector(buf(18 + i * 8, 8), pkt, m, i)
			end
		end
    end

    -- 49 = 17 + 8 * 4(falconigk)
	--18+8*8=82(robinw falcon2.1)
    function InnoBlock2_dissector(buf, pkt, parent, index)

		if is_falcon_igk == true
		then
			local m = parent:add("InnoBlock2[" .. tostring(index) .. "]")
			InnoBlockHeader_dissector(buf(0,17),pkt,m)
			for i=0,7,1 do
				InnoChannelPoint_dissector(buf(17 + i * 4, 4), pkt, m, i)
			end
        elseif is_robinw_generic == true then
			local m = parent:add("InnoCoBlock2[" .. tostring(index) .. "]")
            InnoBlockHeader_dissector(buf(0,10),pkt,m)
            for i=0,15,1 do
                InnoChannelPoint_dissector(buf(10 + i * 4, 4), pkt, m, i)
            end
		else
			local m = parent:add("InnoEnBlock2[" .. tostring(index) .. "]")
			InnoBlockHeader_dissector(buf(0,18),pkt,m)
			for i=0,7,1 do
				InnoChannelPoint_dissector(buf(18 + i * 8, 8), pkt, m, i)
			end
		end
    end

    --[[ ---------------------------------------------------------------------
            struct InnoMessage
            {
                uint32_t size;
                uint32_t src;
                uint64_t id;
                uint32_t level;
                uint32_t code;
                int32_t reserved[4];
                char content[0];
            };
    --]] ---------------------------------------------------------------------

    local field_msg_size = ProtoField.uint32("Innovusion", "size", base.DEC, nil, nil, "size of the whole InnoMessage i.e. size of content + sizeof(InnoMessage)")
    local field_msg_src = ProtoField.uint32("Innovusion", "src", base.DEC, nil, nil, "")
    local field_msg_id = ProtoField.uint64("Innovusion", "id", base.DEC, nil, nil, "")

    local type_msg_level = {
        [0] = "INNO_MESSAGE_LEVEL_FATAL",
        [1] = "INNO_MESSAGE_LEVEL_CRITICAL",
        [2] = "INNO_MESSAGE_LEVEL_ERROR",
        [3] = "INNO_MESSAGE_LEVEL_TEMP",
        [4] = "INNO_MESSAGE_LEVEL_WARNING",
        [5] = "INNO_MESSAGE_LEVEL_DEBUG",
        [6] = "INNO_MESSAGE_LEVEL_INFO",
        [7] = "INNO_MESSAGE_LEVEL_TRACE",
        [8] = "INNO_MESSAGE_LEVEL_DETAIL"
    }
    local field_msg_level = ProtoField.uint32("Innovusion", "level", base.DEC, type_msg_level, nil, "enum InnoMessageLevel")

    local type_msg_code = {
        [0] = "INNO_MESSAGE_CODE_NONE",
        [1] = "INNO_MESSAGE_CODE_LIB_VERSION_MISMATCH",
        [2] = "INNO_MESSAGE_CODE_READ_TIMEOUT",
        [3] = "INNO_MESSAGE_CODE_CANNOT_READ",
        [4] = "INNO_MESSAGE_CODE_BAD_CONFIG_YAML",
        [5] = "INNO_MESSAGE_CODE_OVERHEAT_PROTECTION",
        [6] = "INNO_MESSAGE_CODE_TO_NON_WORKING_MODE",
        [7] = "INNO_MESSAGE_CODE_READ_FILE_END",
        [8] = "INNO_MESSAGE_CODE_RAW_RECORDING_FINISHED",
        [9] = "INNO_MESSAGE_CODE_NEW_START",
        [10001] = "INNO_MESSAGE_CODE_GALVO_MIRROR_CHECK_RESULT",
        [10002] = "INNO_MESSAGE_CODE_MAX_DISTANCE_CHECK_RESULT"
    }
    local field_msg_code = ProtoField.uint32("Innovusion", "code", base.DEC, type_msg_code, nil, "enum InnoMessageCode")

    local field_msg_reserved = ProtoField.none("Innovusion", "reserved", base.DEC, nil, nil, "all 0")

    local field_msg_content = ProtoField.string("Innovusion", "content", base.ASCII, nil, nil, "0 end string")


    --
    -- size : 40+
    --
    function InnoMessage_dissector(buf,pkt,parent)
        local m = parent:add("InnoMessage")

        m:add_le(field_msg_size,buf(0,4))
        m:add_le(field_msg_src,buf(4,4))
        m:add_le(field_msg_id,buf(8,8))
        m:add_le(field_msg_level,buf(16,4))
        m:add_le(field_msg_code,buf(20,4))
        m:add_le(field_msg_reserved,buf(24,4*4))
        m:add_le(field_msg_content,buf(40))
    end


    --[[ ---------------------------------------------------------------------
            /* compact format, 16 + 8 + 2 = 26 bytes per point */
            struct InnoXyzPoint
            {
                float x;
                float y;
                float z;
                float radius;

                uint16_t ts_10us;

                uint16_t scan_id: 9;
                uint16_t in_roi: 2;
                uint16_t facet: 3;
				uint16_t multi_return: 1;
                uint16_t reserved_flags: 1;

                uint32_t is_2nd_return: 1;
                uint32_t scan_idx: 14;
                uint32_t refl: 9;
                uint32_t type: 2;
                uint32_t elongation: 4;
                uint32_t channel: 2;
                uint16_t ring_id;
            };

			/* compact format, 30 bytes per point */
				typedef DEFINE_INNO_COMPACT_STRUCT(InnoEnXyzPoint) {
				  float x;
				  float y;
				  float z;
				  float radius;
				  uint16_t ts_10us;
				  uint16_t scan_id : 10; /* id of the scan line */
				  uint16_t in_roi : 2;
				  uint16_t facet : 3;
				  uint16_t multi_return : 1;    /* multi return mode,true mean the 2nd point*/
				  uint16_t scan_idx : 14;       /* point idx within the scan line */
				  uint16_t type : 2;            /* 0: normal, 1: ground, 2: fog                 */
				  uint16_t reflectance;         /* reflectance, falcon 1-65535,robin 1-4095   */
				  uint16_t intensity;           /* intensity, falcon 1-65535,robin 1-4095   */
				  uint8_t elongation : 7;       /* elongation, unit: 1ns */
				  uint8_t is_2nd_return : 1;
				  uint8_t channel : 3;          /* max 8 channel */
				  uint8_t firing : 1;
				  uint8_t reserved_flags : 4;   /* all 0 */
				  uint16_t ring_id;
				}
    --]] ---------------------------------------------------------------------
	local type_xyz_type = {
			[0] = "normal",
			[1] = "ground",
			[2] = "fog"
	}

	local field_xyz_x = ProtoField.float("Innovusion", "x", nil, "")
	local field_xyz_y = ProtoField.float("Innovusion", "y", nil, "")
	local field_xyz_z = ProtoField.float("Innovusion", "z", nil, "")
	local field_xyz_radius = ProtoField.float("Innovusion", "radius", nil, "")
	local field_xyz_ts_10us = ProtoField.uint16("Innovusion", "ts_10us", base.DEC, nil, nil, "")
	local field_xyz_ring_id = ProtoField.uint16("Innovusion", "ring_id", base.DEC, nil, nil, "")


		-- 2
		field_xyz_scan_id = ProtoField.uint16("Innovusion", "scan_id", base.DEC, nil, 0x01FF, "id of the scan line")
		field_xyz_in_roi = ProtoField.uint16("Innovusion", "in_roi", base.DEC, nil, 0x0600, "")
		field_xyz_facet = ProtoField.uint16("Innovusion", "facet", base.DEC, nil, 0x3800, "")
		field_xyz_multi_return = ProtoField.uint16("Innovusion", "multi_return", base.DEC, nil, 0x4000, "")
		field_xyz_reserved_flags = ProtoField.uint16("Innovusion", "reserved", base.DEC, nil, 0x8000, "all 0")

		-- 4

		field_xyz_is_2nd_return = ProtoField.uint32("Innovusion", "is_2nd_return", base.DEC, nil, 0x00000001, "0: normal, 1: ground, 2: fog")
		field_xyz_scan_idx = ProtoField.uint32("Innovusion", "scan_idx", base.DEC, nil, 0x00007FFE, "point idx within the scan line")
		field_xyz_refl = ProtoField.uint32("Innovusion", "refl", base.DEC, nil, 0x00FF8000, "reflectance, 1-254, 255 means a reflector or intensity, also 1-254 & 255=reflector")
		field_xyz_type = ProtoField.uint32("Innovusion", "level", base.DEC, type_xyz_type, 0x03000000, "")
		field_xyz_elongation = ProtoField.uint32("Innovusion", "elongation", base.DEC, nil, 0x3C000000, "")
		field_xyz_channel = ProtoField.uint32("Innovusion", "channel", base.DEC, nil, 0xC0000000, "")



		field_xyz_scan_id1 = ProtoField.uint16("Innovusion", "scan_id", base.DEC, nil, 0x03FF, "id of the scan line")
		field_xyz_in_roi1 = ProtoField.uint16("Innovusion", "in_roi", base.DEC, nil, 0x0C00, "")
		field_xyz_facet1 = ProtoField.uint16("Innovusion", "facet", base.DEC, nil, 0x7000, "")
		field_xyz_multi_return1 = ProtoField.uint16("Innovusion", "multi_return", base.DEC, nil, 0x8000, "")

		field_xyz_scan_idx1 = ProtoField.uint32("Innovusion", "scan_idx", base.DEC, nil, 0x3FFF, "point idx within the scan line")
		field_xyz_type1 = ProtoField.uint32("Innovusion", "level", base.DEC, type_xyz_type, 0xC000, "")

		field_xyz_refl1 = ProtoField.uint32("Innovusion", "refl", base.DEC, nil, nil, "reflectance, 1-254, 255 means a reflector or intensity, also 1-254 & 255=reflector")
		field_xyz_intensity = ProtoField.uint32("Innovusion", "intensity", base.DEC, nil, nil, "intensity, 1-65535")

		field_xyz_elongation1 = ProtoField.uint32("Innovusion", "elongation", base.DEC, nil, 0x7F, "")
		field_xyz_is_2nd_return1 = ProtoField.uint32("Innovusion", "is_2nd_return", base.DEC, nil, 0x80, "0: normal, 1: ground, 2: fog")

		field_xyz_channel1 = ProtoField.uint32("Innovusion", "channel", base.DEC, nil, 0x7, "")
		field_xyz_firing = ProtoField.uint32("Innovusion", "firing", base.DEC, nil, 0x8, "")
		field_xyz_reserved_flags1 = ProtoField.uint16("Innovusion", "reserved", base.DEC, nil, 0xF0, "all 0")

		field_xyz_ring_id = ProtoField.uint32("Innovusion", "ring_id", base.DEC, nil, nil, "")

    -- size : 26(falconigk)  30(falcon2.1 robinw)

    function InnoXyzPoint_dissector(buf,pkt,parent)
        local m = parent:add("InnoXyzPoint")

        if is_falcon_igk == true
		then
			m:add_le(field_xyz_x,buf(0,4))
			m:add_le(field_xyz_y,buf(4,4))
			m:add_le(field_xyz_z,buf(8,4))
			m:add_le(field_xyz_radius,buf(12,4))

			m:add_le(field_xyz_ts_10us,buf(16,2))

			--
			m:add_le(field_xyz_scan_id,buf(18,2))
			m:add_le(field_xyz_in_roi,buf(18,2))
			m:add_le(field_xyz_facet,buf(18,2))
			m:add_le(field_xyz_multi_return,buf(18,2))
			m:add_le(field_xyz_reserved_flags,buf(18,2))

			--
			m:add_le(field_xyz_is_2nd_return,buf(20,4))
			m:add_le(field_xyz_scan_idx,buf(20,4))
			m:add_le(field_xyz_refl,buf(20,4))
			m:add_le(field_xyz_type,buf(20,4))
			m:add_le(field_xyz_elongation,buf(20,4))
			m:add_le(field_xyz_channel,buf(20,4))


			m:add_le(field_xyz_ring_id,buf(24,2))
		else
			m:add_le(field_xyz_x,buf(0,4))
			m:add_le(field_xyz_y,buf(4,4))
			m:add_le(field_xyz_z,buf(8,4))
			m:add_le(field_xyz_radius,buf(12,4))
			m:add_le(field_xyz_ts_10us,buf(16,2))

			m:add_le(field_xyz_scan_id1,buf(18,2))
			m:add_le(field_xyz_in_roi1,buf(18,2))
			m:add_le(field_xyz_facet1,buf(18,2))
			m:add_le(field_xyz_multi_return1,buf(18,2))

			m:add_le(field_xyz_scan_idx1,buf(20,2))
			m:add_le(field_xyz_type1,buf(20,2))

			m:add_le(field_xyz_refl1,buf(22,2))
			m:add_le(field_xyz_intensity,buf(24,2))

			m:add_le(field_xyz_elongation1,buf(26,1))
			m:add_le(field_xyz_is_2nd_return1,buf(26,1))

			m:add_le(field_xyz_channel1,buf(27,1))
			m:add_le(field_xyz_firing,buf(27,1))
			m:add_le(field_xyz_reserved_flags1,buf(27,1))

			m:add_le(field_xyz_ring_id,buf(28,2))
		end

    end


    --[[ ---------------------------------------------------------------------
            struct InnoCommonVersion
            {
                uint16_t magic_number;

                uint8_t major_version;
                uint8_t minor_version;

                uint16_t fw_sequence;
            };
    --]] ---------------------------------------------------------------------

    local type_magic_number = {
        [0x176A] = "InnoDataPacket",
        [0x186B] = "InnoStatusPacket"
    }
    local field_magic_number = ProtoField.uint16("Innovusion", "magic_number", base.HEX, type_magic_number)
    local field_major_version = ProtoField.uint8("Innovusion","major_version",base.DEC)
    local field_minor_version = ProtoField.uint8("Innovusion","minor_version",base.DEC)
    local field_fw_sequence = ProtoField.uint16("Innovusion","fw_sequence",base.DEC)

    --
    -- size : 6
    --
    function InnoCommonVersion_dissector(buf,pkt,parent)
        local t = parent:add("InnoCommonVersion")
        t:add_le(field_magic_number,buf(0,2))
        t:add_le(field_major_version,buf(2,1))
        t:add_le(field_minor_version,buf(3,1))
        t:add_le(field_fw_sequence,buf(4,2))
    end



    --[[ ---------------------------------------------------------------------
            struct InnoCommonHeader
            {
                /* 6 bytes */
                InnoCommonVersion version;

                /* 4 bytes, cover every thing except checksum */
                uint32_t checksum;

                /* 4 bytes */
                uint32_t size;

                /* 2 bytes */
                uint16_t source_id :4;           /* up to 16 different LiDAR source */
                uint16_t timestamp_sync_type :4; /* enum InnoTimestampSyncType      */
                uint16_t reserved :8;

                /* 8 bytes */
                InnoTimestampUs ts_start_us; /* epoch time of start of frame, in micro-sec */

                /* 2 bytes */
                uint8_t lidar_mode;        /* enum InnoLidarMode    */
                uint8_t lidar_status;      /* enum InnoLidarStatus  */
            };
    --]] ---------------------------------------------------------------------

    local field_checksum = ProtoField.uint32("Innovusion","checksum",base.HEX)
    local field_size = ProtoField.uint32("Innovusion","size",base.DEC)
    local field_source_id = ProtoField.uint16("Innovusion","source_id",base.DEC, nil, 0x000F, "up to 16 different LiDAR source")

    local type_time_sync = {
        [0] = "INNO_TIME_SYNC_TYPE_NONE",
        [1] = "INNO_TIME_SYNC_TYPE_RECORDED",
        [2] = "INNO_TIME_SYNC_TYPE_HOST",
        [3] = "INNO_TIME_SYNC_TYPE_GPS_INIT",
        [4] = "INNO_TIME_SYNC_TYPE_GPS_LOCKED",
        [5] = "INNO_TIME_SYNC_TYPE_GPS_UNLOCKED",
        [6] = "INNO_TIME_SYNC_TYPE_PTP_INIT",
        [7] = "INNO_TIME_SYNC_TYPE_PTP_LOCKED",
        [8] = "INNO_TIME_SYNC_TYPE_PTP_UNLOCKED",
        [9] = "INNO_TIME_SYNC_TYPE_FILE_INIT",
        [10] = "INNO_TIME_SYNC_TYPE_NTP_INIT",
        [11] = "INNO_TIME_SYNC_TYPE_NTP_LOCKED",
        [12] = "INNO_TIME_SYNC_TYPE_NTP_UNLOCKED",
        [13] = "INNO_TIME_SYNC_TYPE_GPS_LOST",
        [14] = "INNO_TIME_SYNC_TYPE_PTP_LOST",
        [15] = "INNO_TIME_SYNC_TYPE_NTP_LOST"
    }
    local field_timestamp_sync_type = ProtoField.uint16("Innovusion","timestamp_sync_type",base.DEC, type_time_sync, 0x00F0, "enum InnoTimestampSyncType")

	local lidar_type = {
		[0] = "falconi/g/k",
		[1] = "robinw",
		[2] = "robinE",
		[3] = "falcon2.1",
		[4] = "falcon3"
	}
    local field_header_reserved = ProtoField.uint16("Innovusion","reserved/lidar_type",base.DEC, lidar_type, 0xFF00, "")

    local field_ts_start_us = ProtoField.double("Innovusion","ts_start_us", nil, "epoch time of start of frame, in micro-sec")

    local type_lidar_mode = {
        [0] = "INNO_LIDAR_MODE_NONE",
        [1] = "INNO_LIDAR_MODE_SLEEP",
        [2] = "INNO_LIDAR_MODE_STANDBY",
        [3] = "INNO_LIDAR_MODE_WORK_NORMAL",
        [4] = "INNO_LIDAR_MODE_WORK_SHORT_RANGE",
        [5] = "INNO_LIDAR_MODE_WORK_CALIBRATION",
        [6] = "INNO_LIDAR_MODE_PROTECTION",
        [7] = "INNO_LIDAR_MODE_WORK_QUIET",
        [8] = "INNO_LIDAR_MODE_WORK_INTERNAL_1"}
    local field_lidar_mode = ProtoField.uint8("Innovusion", "lidar_mode", base.DEC, type_lidar_mode)

    local type_lidar_status = {
        [0] = "INNO_LIDAR_STATUS_NONE",
        [1] = "INNO_LIDAR_STATUS_TRANSITION",
        [2] = "INNO_LIDAR_STATUS_NORMAL",
        [3] = "INNO_LIDAR_STATUS_FAILED" }
    local field_lidar_status = ProtoField.uint8("Innovusion", "lidar_status", base.DEC, type_lidar_status)


    --
    -- InnoCommonHeader
    -- size : 26
    --
    function InnoCommonHeader_dissector(buf,pkt,parent)
        local h = parent:add("InnoCommonHeader")

        InnoCommonVersion_dissector(buf(0,6),pkt,h)

        h:add_le(field_checksum,buf(6,4))
        h:add_le(field_size,buf(10,4))

        h:add_le(field_source_id,buf(14,2))
        h:add_le(field_timestamp_sync_type,buf(14,2))
        h:add_le(field_header_reserved,buf(14,2))

        -- ts
        local s = h:add_le(field_ts_start_us, buf(16,8))

        -- ts_start_us: 1.64102045378074e+15
        local ss = string.sub(s.text, 14)
        local ss_len = string.len(ss)
        if ( ss_len >= 15 ) then
            -- 1.64102045378074e+15
            local sec_str = string.sub(ss, 1, 1) .. string.sub(ss, 3, 3 + 8)
            local usec_str = "00000"

            local us_len = ss_len - 15
            if (us_len > 0) then
                usec_str = string.sub(ss, 12, 12 + us_len - 1) .. string.rep("0", 5 - us_len)
            end

            local str_ts = os.date("!%Y-%m-%d %H:%M:%S", tonumber(sec_str))
            s:append_text(" [utc: " .. str_ts .. ":" .. usec_str .. "0]")
        end

        -- wireshark bugs, le_float() return nil
        -- local ts = buf(16,8):le_float()
        -- local sec = math.floor(ts)
        -- local usec = math.floor((ts - sec) * 100000)

        h:add_le(field_lidar_mode,buf(24,1))
        h:add_le(field_lidar_status,buf(25,1))
    end


    --[[ ---------------------------------------------------------------------
            struct InnoDataPacket
            {
                InnoCommonHeader common;

                /* 12 bytes */
                uint64_t idx;         /* frame index, start from 0                     */
                uint16_t sub_idx;     /* sub-frame index, start from 0 for every frame */
                uint16_t sub_seq;     /* sequence within a sub-frame                   */

                /* 10 byte */
                /* type in enum InnoItemType, each type uses independent global idx */
                uint32_t type :8;
                uint32_t item_number :24;        /* max 4 * 1024 * 1024               */
                uint16_t item_size;              /* max 65535, 0 means variable size  */
                uint32_t topic;                  /* reserved                          */

                /* 2 bytes */
                uint16_t scanner_direction :1; /* 0: top->bottom, 1: bottom->top          */
                uint16_t use_reflectance   :1; /* 0: intensity mode, 1: reflectance mode  */
                uint16_t multi_return_mode :3; /* ... */
                uint16_t confidence_level  :2; /* 0: no confidence, 3: higest             */
                uint16_t is_last_sub_frame :1; /* 1: the last sub frame of a frame        */
                uint16_t is_last_sequence  :1; /* 1: the last piece of a sub frame        */
                uint16_t has_tail :1;          /* has additional tail struct after points */
                uint16_t reserved_flag :6;     /* all 0 */

                /* 4 bytes */
                int16_t roi_h_angle;           /* configured ROI in InnoAngleUnit */
                int16_t roi_v_angle;

                union {
                    char c[0];
                    InnoBlock1 inno_block1s[0];
                    InnoBlock2 inno_block2s[0];
                    InnoMessage messages[0];
                    InnoXyzPoint xyz_points[0];
                };
            };


			typedef DEFINE_INNO_COMPACT_STRUCT(InnoDataPacket) {
				  InnoCommonHeader common;

				  /* 12 bytes */
				  uint64_t idx;         /* frame index, start from 0                     */
				  uint16_t sub_idx;     /* sub-frame index, start from 0 for every frame */
				  uint16_t sub_seq;     /* sequence within a sub-frame                   */

				  /* 10 byte */
				  /* type in enum InnoItemType, each type uses independent global idx */
				  uint32_t type :8;
				  uint32_t item_number :24;        /* max 4 * 1024 * 1024               */
				  uint16_t item_size;              /* max 65535, 0 means variable size  */
				  uint32_t topic;                  /* reserved                          */

				  /* 2 bytes */
				  uint16_t scanner_direction :1; /* 0: top->bottom, 1: bottom->top          */
				  uint16_t use_reflectance   :1; /* 0: intensity mode, 1: reflectance mode  */
				  uint16_t multi_return_mode :3; /* ... */
				  uint16_t confidence_level  :2; /* 0: no confidence, 3: higest             */
				  uint16_t is_last_sub_frame :1; /* 1: the last sub frame of a frame        */
				  uint16_t is_last_sequence  :1; /* 1: the last piece of a sub frame        */
				  uint16_t has_tail :1;          /* has additional tail struct after points */
				  uint16_t frame_sync_locked :1; /* 1: frame sync has locked                */
				  uint16_t is_first_sub_frame :1; /* 1: the first sub frame of a frame      */
				  uint16_t last_four_channel :1;
				  uint16_t reserved_flag :3;     /* all 0 */

				  /* 4 bytes */
				  int16_t roi_h_angle;           /* configured ROI in InnoAngleUnit */
				  int16_t roi_v_angle;
				  uint32_t extend_reserved[4];  /* add more extend reserved area 16 byte */
				// MSVC compiler does not support multi-dimensional flexible arrays.
				# if !defined(_MSC_VER)
				  union {
					char payload[0];
					InnoBlock1 inno_block1s[0];
					InnoBlock2 inno_block2s[0];
					InnoMessage messages[0];
					InnoXyzPoint xyz_points[0];
						// Robin & Falcon2.1
					InnoEnBlock1 inno_en_block1s[0];
					InnoEnBlock2 inno_en_block2s[0];
					InnoEnXyzPoint en_xyz_points[0];
				  }
				  #else
				char payload[0];
				#endif
					}

            /*
            * Main data packet definition
            * 26 + 12 + 10 + 2 + 4 + 16 = 70 bytes, max overhead is 70/1472 = 4.7%,
            */
            typedef DEFINE_INNO_COMPACT_STRUCT(InnoDataPacket) {
            InnoCommonHeader common;

            /* 12 bytes */
            uint64_t idx;     /* frame index, start from 0                     */
            uint16_t sub_idx; /* sub-frame index, start from 0 for every frame */
            uint16_t sub_seq; /* sequence within a sub-frame                   */

            /* 10 byte */
            /* type in enum InnoItemType, each type uses independent global idx */
            uint32_t type : 8;
            uint32_t item_number : 24; /* max 4 * 1024 * 1024               */
            uint16_t item_size;        /* max 65535, 0 means variable size  */
            uint32_t topic;            /* reserved                          */

            /* 2 bytes */
            uint16_t scanner_direction : 1;  /* 0: top->bottom, 1: bottom->top          */
            uint16_t use_reflectance : 1;    /* 0: intensity mode, 1: reflectance mode  */
            uint16_t multi_return_mode : 3;  /* ... */
            uint16_t confidence_level : 2;   /* 0: no confidence, 3: higest             */
            uint16_t is_last_sub_frame : 1;  /* 1: the last sub frame of a frame        */
            uint16_t is_last_sequence : 1;   /* 1: the last piece of a sub frame        */
            uint16_t has_tail : 1;           /* has additional tail struct after points */
            uint16_t frame_sync_locked : 1;  /* 1: frame sync has locked                */
            uint16_t is_first_sub_frame : 1; /* 1: the first sub frame of a frame      */
            uint16_t last_four_channel : 1;
            uint16_t reserved_flag : 3; /* all 0 */

            /* 4 bytes */
            int16_t roi_h_angle; /* configured ROI in InnoAngleUnit */
            int16_t roi_v_angle;
            uint32_t extend_reserved[4]; /* add more extend reserved area 16 byte */
            // MSVC compiler does not support multi-dimensional flexible arrays.
            #if !defined(_MSC_VER)
            union {
                char payload[0];
                InnoBlock1 inno_block1s[0];
                InnoBlock2 inno_block2s[0];
                InnoMessage messages[0];
                InnoXyzPoint xyz_points[0];
                // Robin & Falcon2.1
                InnoEnBlock1 inno_en_block1s[0];
                InnoEnBlock2 inno_en_block2s[0];
                InnoCoBlock1 inno_co_block1s[0];
                InnoCoBlock2 inno_co_block2s[0];
                InnoEnXyzPoint en_xyz_points[0];
                InnoAngleHVTable inno_anglehv_table[0];
            };
            #else
            char payload[0];
            #endif
            }
            InnoDataPacket;
    --]] ---------------------------------------------------------------------
	local type_item_type = {
        [0] = "INNO_ITEM_TYPE_NONE",
        [1] = "INNO_ITEM_TYPE_SPHERE_POINTCLOUD",
        [2] = "INNO_ITEM_TYPE_MESSAGE",
        [3] = "INNO_ITEM_TYPE_MESSAGE_LOG",
        [4] = "INNO_ITEM_TYPE_XYZ_POINTCLOUD",
		[5] = "INNO_ROBINE_ITEM_TYPE_SPHERE_POINTCLOUD",
		[6] = "INNO_ROBINE_ITEM_TYPE_XYZ_POINTCLOUD",
		[7] = "INNO_ROBINW_ITEM_TYPE_SPHERE_POINTCLOUD",
		[8] = "INNO_ROBINW_ITEM_TYPE_XYZ_POINTCLOUD",
		[9] = "INNO_FALCONII_DOT_1_ITEM_TYPE_SPHERE_POINTCLOUD",
		[10] = "INNO_FALCONII_DOT_1_ITEM_TYPE_XYZ_POINTCLOUD",
		[11] = "INNO_FALCONIII_ITEM_TYPE_SPHERE_POINTCLOUD",
		[12] = "INNO_FALCONIII_ITEM_TYPE_XYZ_POINTCLOUD",
        [13] = "INNO_ROBINW_ITEM_TYPE_COMPACT_POINTCLOUD",
        [14] = "INNO_ROBINELITE_ITEM_TYPE_SPHERE_POINTCLOUD",
        [15] = "INNO_ROBINELITE_ITEM_TYPE_XYZ_POINTCLOUD",
        [16] = "INNO_ROBINELITE_ITEM_TYPE_COMPACT_POINTCLOUD"
    }
	local type_scanner_direction = {
        [0] = "top->bottom",
        [1] = "bottom->top" }
    local type_use_reflectance = {
        [0] = "intensity mode",
        [1] = "reflectance mode" }

	-- frame
    local field_frame_index = ProtoField.uint64("Innovusion", "frame_index(idx)", base.DEC, nil, nil, "frame index, start from 0")
    local field_frame_sub_idx = ProtoField.uint16("Innovusion", "sub_idx", base.DEC, nil, nil, "sub-frame index, start from 0 for every frame")
    local field_frame_sub_seq = ProtoField.uint16("Innovusion", "sub_seq", base.DEC, nil, nil, "sequence within a sub-frame")

    -- item

    local field_item_type = ProtoField.uint32("Innovusion", "item_type(type)", base.DEC, type_item_type, 0x000000FF, "enum InnoItemType, each type uses independent global idx")
    local field_item_number = ProtoField.uint32("Innovusion", "item_number", base.DEC, nil, 0xFFFFFF00, "max 4 * 1024 * 1024")

    local field_item_size = ProtoField.uint16("Innovusion", "item_size", base.DEC, nil, nil, "max 65535, 0 means variable size")
    local field_topic = ProtoField.uint32("Innovusion", "topic", base.DEC, nil, nil, "reserved")

    -- flags

    local field_scanner_direction = ProtoField.uint16("Innovusion", "scanner_direction", base.DEC, type_scanner_direction, 0x0001, "")


    local field_use_reflectance = ProtoField.uint16("Innovusion", "use_reflectance", base.DEC, type_use_reflectance, 0x0002, "")

    local field_multi_return_mode = ProtoField.uint16("Innovusion", "multi_return_mode", base.DEC, nil, 0x001C, "")
    local field_confidence_level = ProtoField.uint16("Innovusion", "confidence_level", base.DEC, nil, 0x0060, "")
    local field_is_last_sub_frame = ProtoField.uint16("Innovusion", "is_last_sub_frame", base.DEC, nil, 0x0080, "the last sub frame of a frame")
    local field_is_last_sequence = ProtoField.uint16("Innovusion", "is_last_sequence", base.DEC, nil, 0x0100, "the last piece of a sub frame")
    local field_has_tail = ProtoField.uint16("Innovusion", "has_tail", base.DEC, nil, 0x0200, "has additional tail struct after points")
	-- roi
	local field_roi_h_angle = ProtoField.uint16("Innovusion", "roi_h_angle", base.DEC, nil, nil, "configured ROI in InnoAngleUnit")
	local field_roi_v_angle = ProtoField.uint16("Innovusion", "roi_v_angle", base.DEC, nil, nil, "configured ROI in InnoAngleUnit")

	    field_reserved_flag = ProtoField.uint16("Innovusion", "reserved_flag", base.DEC, nil, 0xFC00, "")
		field_frame_sync_locked = ProtoField.uint16("Innovusion", "frame_sync_locked", base.DEC, nil, 0x0400, "")
		field_is_first_sub_frame = ProtoField.uint16("Innovusion", "is_first_sub_frame", base.DEC, nil, 0x0800, "")
		field_last_four_channel = ProtoField.uint16("Innovusion", "last_four_channel", base.DEC, nil, 0x1000, "")
		field_reserved_flag1 = ProtoField.uint16("Innovusion", "reserved_flag", base.DEC, nil, 0xE000, "")




    local field_infault_0 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_OTHER", base.DEC, nil, 0x01, "")
    local field_infault_1 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_POWER_LOW", base.DEC, nil, 0x02, "")
    local field_infault_2 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_POWER_HIGH", base.DEC, nil, 0x04, "")
    local field_infault_3 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_WINDOW_BLOCKAGE1", base.DEC, nil, 0x08, "")
    local field_infault_4 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_WINDOW_BLOCKAGE2", base.DEC, nil, 0x10, "")
    local field_infault_5 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_WINDOW_BLOCKAGE3", base.DEC, nil, 0x20, "")
    local field_infault_6 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_WINDOW_BLOCKAGE4", base.DEC, nil, 0x40, "")
    local field_infault_7 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_LASER_INTERLOCK", base.DEC, nil, 0x80, "")
    local field_infault_8 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_COMM_LASER", base.DEC, nil, 0x1, "")
    local field_infault_9 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_LASER", base.DEC, nil, 0x2, "")
    local field_infault_10 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_COMM_DSP", base.DEC, nil, 0x4, "")
    local field_infault_11 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_CANFD_DSP", base.DEC, nil, 0x8, "")
    local field_infault_12 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_DSP", base.DEC, nil, 0x10, "")
    local field_infault_13 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_POLYGON_CONTROL", base.DEC, nil, 0x20, "")
    local field_infault_14 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_POLYGON_SENSOR", base.DEC, nil, 0x40, "")
    local field_infault_15 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_GALVO_CONTROL", base.DEC, nil, 0x80, "")
    local field_infault_16 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_GALVO_SENSOR", base.DEC, nil, 0x1, "")
    local field_infault_17 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_OPTIC1", base.DEC, nil, 0x2, "")
    local field_infault_18 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_OPTIC2", base.DEC, nil, 0x4, "")
    local field_infault_19 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_IIC_DSP", base.DEC, nil, 0x8, "")
    local field_infault_20 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_IIC_SOC", base.DEC, nil, 0x10, "")
    local field_infault_21 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_DSP_EXTWD", base.DEC, nil, 0x20, "")
    local field_infault_22 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_DBTEMP", base.DEC, nil, 0x40, "")
    local field_infault_23 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_CHIPTEMP", base.DEC, nil, 0x80, "")
    local field_infault_24 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_HUMIDITY", base.DEC, nil, 0x1, "")
    local field_infault_25 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_COMM_ADC", base.DEC, nil, 0x2, "")
    local field_infault_26 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_FPGACLOCK", base.DEC, nil, 0x4, "")
    local field_infault_27 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_SOC", base.DEC, nil, 0x8, "")
    local field_infault_28 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_SOC_EXTWD", base.DEC, nil, 0x10, "")
    local field_infault_29 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_RAWDATA_STREAM", base.DEC, nil, 0x20, "")
    local field_infault_30 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_POLYGON_TO", base.DEC, nil, 0x40, "")
    local field_infault_31 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_GALVO_TO", base.DEC, nil, 0x80, "")
    local field_infault_32 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_TRIGGER_TO", base.DEC, nil, 0x1, "")
    local field_infault_33 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_POWSUPL1", base.DEC, nil, 0x2, "")
    local field_infault_34 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_POWSUPL2", base.DEC, nil, 0x4, "")
    local field_infault_35 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_LPDDR4", base.DEC, nil, 0x8, "")
    local field_infault_36 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_FLASH", base.DEC, nil, 0x10, "")
    local field_infault_37 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_NETWORK1", base.DEC, nil, 0x20, "")
    local field_infault_38 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_NETWORK2", base.DEC, nil, 0x40, "")
    local field_infault_39 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_OVERHEAT1", base.DEC, nil, 0x80, "")
    local field_infault_40 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_OVERHEAT2", base.DEC, nil, 0x1, "")
    local field_infault_41 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_OVERHEAT3", base.DEC, nil, 0x2, "")
    local field_infault_42 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_CONFIG1", base.DEC, nil, 0x4, "")
    local field_infault_43 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_CONFIG2", base.DEC, nil, 0x8, "")
    local field_infault_44 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_ASSERT_FAILURE", base.DEC, nil, 0x10, "")
    local field_infault_45 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_CPULOAD_HIGH", base.DEC, nil, 0x20, "")
    local field_infault_46 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_LATENCY_LONG", base.DEC, nil, 0x40, "")
    local field_infault_47 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_RESERVED07", base.DEC, nil, 0x80, "")
    local field_infault_48 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_RESERVED08", base.DEC, nil, 0x1, "")
    local field_infault_49 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_EXCESSIVE_NOISE", base.DEC, nil, 0x2, "")
    local field_infault_50 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_DATA_DROP1", base.DEC, nil, 0x4, "")
    local field_infault_51 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_DATA_DROP2", base.DEC, nil, 0x8, "")
    local field_infault_52 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_DATA_DROP3", base.DEC, nil, 0x10, "")
    local field_infault_53 = ProtoField.uint8("Innovusion", "INNO_LIDAR_TEMPHIGH_INHIBIT", base.DEC, nil, 0x20, "")
    local field_infault_54 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_RESERVED10", base.DEC, nil, 0x40, "")
    local field_infault_55 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_REFINTENSITY", base.DEC, nil, 0x80, "")
    local field_infault_56 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_REPROGRAMMING", base.DEC, nil, 0x1, "")
    local field_infault_57 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_GALVO_MIRROR", base.DEC, nil, 0x2, "")
    local field_infault_58 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_MAX_DISTANCE", base.DEC, nil, 0x4, "")
    local field_infault_59 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_GALVO_OFFSET", base.DEC, nil, 0x8, "")
    local field_infault_60 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_RESERVED15", base.DEC, nil, 0x10, "")
    local field_infault_61 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_RESERVED16", base.DEC, nil, 0x20, "")
    local field_infault_62 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_RESERVED17", base.DEC, nil, 0x40, "")
    local field_infault_63 = ProtoField.uint8("Innovusion", "INNO_LIDAR_IN_FAULT_RESERVED18", base.DEC, nil, 0x80, "")
    local field_extended_faults = ProtoField.uint32("Innovusion", "extended_faults", base.DEC, nil, nil, "")

	--64+32=96  96/8=12bytes
    function InnoStatusInFaults_dissector(buf,pkt,parent)
        local h = parent:add("InnoStatusInFaults")

        h:add_le(field_infault_0,buf(0,1))
        h:add_le(field_infault_1,buf(0,1))
        h:add_le(field_infault_2,buf(0,1))
        h:add_le(field_infault_3,buf(0,1))
        h:add_le(field_infault_4,buf(0,1))
        h:add_le(field_infault_5,buf(0,1))
        h:add_le(field_infault_6,buf(0,1))
        h:add_le(field_infault_7,buf(0,1))
        h:add_le(field_infault_8,buf(1,1))
        h:add_le(field_infault_9,buf(1,1))
        h:add_le(field_infault_10,buf(1,1))
        h:add_le(field_infault_11,buf(1,1))
        h:add_le(field_infault_12,buf(1,1))
        h:add_le(field_infault_13,buf(1,1))
        h:add_le(field_infault_14,buf(1,1))
        h:add_le(field_infault_15,buf(1,1))
        h:add_le(field_infault_16,buf(2,1))
        h:add_le(field_infault_17,buf(2,1))
        h:add_le(field_infault_18,buf(2,1))
        h:add_le(field_infault_19,buf(2,1))
        h:add_le(field_infault_20,buf(2,1))
        h:add_le(field_infault_21,buf(2,1))
        h:add_le(field_infault_22,buf(2,1))
        h:add_le(field_infault_23,buf(2,1))
        h:add_le(field_infault_24,buf(3,1))
        h:add_le(field_infault_25,buf(3,1))
        h:add_le(field_infault_26,buf(3,1))
        h:add_le(field_infault_27,buf(3,1))
        h:add_le(field_infault_28,buf(3,1))
        h:add_le(field_infault_29,buf(3,1))
        h:add_le(field_infault_30,buf(3,1))
        h:add_le(field_infault_31,buf(3,1))
        h:add_le(field_infault_32,buf(4,1))
        h:add_le(field_infault_33,buf(4,1))
        h:add_le(field_infault_34,buf(4,1))
        h:add_le(field_infault_35,buf(4,1))
        h:add_le(field_infault_36,buf(4,1))
        h:add_le(field_infault_37,buf(4,1))
        h:add_le(field_infault_38,buf(4,1))
        h:add_le(field_infault_39,buf(4,1))
        h:add_le(field_infault_40,buf(5,1))
        h:add_le(field_infault_41,buf(5,1))
        h:add_le(field_infault_42,buf(5,1))
        h:add_le(field_infault_43,buf(5,1))
        h:add_le(field_infault_44,buf(5,1))
        h:add_le(field_infault_45,buf(5,1))
        h:add_le(field_infault_46,buf(5,1))
        h:add_le(field_infault_47,buf(5,1))
        h:add_le(field_infault_48,buf(6,1))
        h:add_le(field_infault_49,buf(6,1))
        h:add_le(field_infault_50,buf(6,1))
        h:add_le(field_infault_51,buf(6,1))
        h:add_le(field_infault_52,buf(6,1))
        h:add_le(field_infault_53,buf(6,1))
        h:add_le(field_infault_54,buf(6,1))
        h:add_le(field_infault_55,buf(6,1))
        h:add_le(field_infault_56,buf(7,1))
        h:add_le(field_infault_57,buf(7,1))
        h:add_le(field_infault_58,buf(7,1))
        h:add_le(field_infault_59,buf(7,1))
        h:add_le(field_infault_60,buf(7,1))
        h:add_le(field_infault_61,buf(7,1))
        h:add_le(field_infault_62,buf(7,1))
        h:add_le(field_infault_63,buf(7,1))
        h:add_le(field_extended_faults,buf(8,4))
    end

    -- DEFINE_INNO_COMPACT_STRUCT(InnoStatusCounters) {
    --     uint64_t point_data_packet_sent;
    --     uint64_t point_sent;
    --     uint64_t message_packet_sent;
    --     uint64_t raw_data_read;
    --     uint64_t total_frame;
    --     uint64_t total_polygon_rotation;
    --     uint64_t total_polygon_facet;
    --     uint32_t power_up_time_in_second;
    --     uint32_t process_up_time_in_second;
    --     uint32_t lose_ptp_sync;
    --     uint32_t bad_data[4];
    --     uint32_t data_drop[8];
    --     uint32_t in_signals[8];
    --     uint16_t latency_10us_average[6];
    --     uint16_t latency_10us_variation[6];
    --     uint16_t latency_10us_max[6];
    --     uint32_t big_latency_frame;
    --     uint32_t bad_frame;
    --     uint32_t big_gap_frame;
    --     uint32_t small_gap_frame;
    --     uint16_t cpu_percentage;
    --     uint16_t mem_percentage;
    --     uint16_t motor[5];  /* std,min,max1,max2 */
    --     uint16_t galvo[5];  /* std,min,max1,max2 */
    --     uint16_t netstat_rx_speed_kBps;
    --     uint16_t netstat_tx_speed_kBps;
    --     uint16_t netstat_rx_drop;
    --     uint16_t netstat_tx_drop;
    --     uint16_t netstat_rx_err;
    --     uint16_t netstat_tx_err;
    --     uint16_t sys_cpu_percentage[4];
    --     uint32_t lifelong_uptime;
    --     uint32_t reserved[18];
    --   };

    field_point_data_packet_sent = ProtoField.uint64("Innovusion", "point_data_packet_sent", base.DEC, nil, nil, "")
    field_point_sent = ProtoField.uint64("Innovusion", "point_sent", base.DEC, nil, nil, "")
    field_message_packet_sent = ProtoField.uint64("Innovusion", "message_packet_sent", base.DEC, nil, nil, "")
    field_raw_data_read = ProtoField.uint64("Innovusion", "raw_data_read", base.DEC, nil, nil, "")
    field_total_frame = ProtoField.uint64("Innovusion", "total_frame", base.DEC, nil, nil, "")
    field_total_polygon_rotation = ProtoField.uint64("Innovusion", "total_polygon_rotation", base.DEC, nil, nil, "")
    field_total_polygon_facet = ProtoField.uint64("Innovusion", "total_polygon_facet", base.DEC, nil, nil, "")
    field_power_up_time_in_second = ProtoField.uint32("Innovusion", "power_up_time_in_second", base.DEC, nil, nil, "")
    field_process_up_time_in_second = ProtoField.uint32("Innovusion", "process_up_time_in_second", base.DEC, nil, nil, "")
    field_lose_ptp_sync = ProtoField.uint32("Innovusion", "lose_ptp_sync", base.DEC, nil, nil, "")
    field_bad_data0 = ProtoField.uint32("Innovusion", "bad_data0", base.DEC, nil, nil, "")
    field_bad_data1 = ProtoField.uint32("Innovusion", "bad_data1", base.DEC, nil, nil, "")
    field_bad_data2 = ProtoField.uint32("Innovusion", "bad_data2", base.DEC, nil, nil, "")
    field_bad_data3 = ProtoField.uint32("Innovusion", "bad_data3", base.DEC, nil, nil, "")
    field_data_drop0 = ProtoField.uint32("Innovusion", "data_drop0", base.DEC, nil, nil, "")
    field_data_drop1 = ProtoField.uint32("Innovusion", "data_drop1", base.DEC, nil, nil, "")
    field_data_drop2 = ProtoField.uint32("Innovusion", "data_drop2", base.DEC, nil, nil, "")
    field_data_drop3 = ProtoField.uint32("Innovusion", "data_drop3", base.DEC, nil, nil, "")
    field_data_drop4 = ProtoField.uint32("Innovusion", "data_drop4", base.DEC, nil, nil, "")
    field_data_drop5 = ProtoField.uint32("Innovusion", "data_drop5", base.DEC, nil, nil, "")
    field_data_drop6 = ProtoField.uint32("Innovusion", "data_drop6", base.DEC, nil, nil, "")
    field_data_drop7 = ProtoField.uint32("Innovusion", "data_drop7", base.DEC, nil, nil, "")
    field_in_signals0 = ProtoField.uint32("Innovusion", "in_signals0", base.DEC, nil, nil, "")
    field_in_signals1 = ProtoField.uint32("Innovusion", "in_signals1", base.DEC, nil, nil, "")
    field_in_signals2 = ProtoField.uint32("Innovusion", "in_signals2", base.DEC, nil, nil, "")
    field_in_signals3 = ProtoField.uint32("Innovusion", "in_signals3", base.DEC, nil, nil, "")
    field_in_signals4 = ProtoField.uint32("Innovusion", "in_signals4", base.DEC, nil, nil, "")
    field_in_signals5 = ProtoField.uint32("Innovusion", "in_signals5", base.DEC, nil, nil, "")
    field_in_signals6 = ProtoField.uint32("Innovusion", "in_signals6", base.DEC, nil, nil, "")
    field_in_signals7 = ProtoField.uint32("Innovusion", "in_signals7", base.DEC, nil, nil, "")
    field_latency_10us_average0 = ProtoField.uint16("Innovusion", "latency_10us_average0", base.DEC, nil, nil, "")
    field_latency_10us_average1 = ProtoField.uint16("Innovusion", "latency_10us_average1", base.DEC, nil, nil, "")
    field_latency_10us_average2 = ProtoField.uint16("Innovusion", "latency_10us_average2", base.DEC, nil, nil, "")
    field_latency_10us_average3 = ProtoField.uint16("Innovusion", "latency_10us_average3", base.DEC, nil, nil, "")
    field_latency_10us_average4 = ProtoField.uint16("Innovusion", "latency_10us_average4", base.DEC, nil, nil, "")
    field_latency_10us_average5 = ProtoField.uint16("Innovusion", "latency_10us_average5", base.DEC, nil, nil, "")
    field_latency_10us_variation0 = ProtoField.uint16("Innovusion", "latency_10us_variation0", base.DEC, nil, nil, "")
    field_latency_10us_variation1 = ProtoField.uint16("Innovusion", "latency_10us_variation1", base.DEC, nil, nil, "")
    field_latency_10us_variation2 = ProtoField.uint16("Innovusion", "latency_10us_variation2", base.DEC, nil, nil, "")
    field_latency_10us_variation3 = ProtoField.uint16("Innovusion", "latency_10us_variation3", base.DEC, nil, nil, "")
    field_latency_10us_variation4 = ProtoField.uint16("Innovusion", "latency_10us_variation4", base.DEC, nil, nil, "")
    field_latency_10us_variation5 = ProtoField.uint16("Innovusion", "latency_10us_variation5", base.DEC, nil, nil, "")
    field_latency_10us_max0 = ProtoField.uint16("Innovusion", "latency_10us_max0", base.DEC, nil, nil, "")
    field_latency_10us_max1 = ProtoField.uint16("Innovusion", "latency_10us_max1", base.DEC, nil, nil, "")
    field_latency_10us_max2 = ProtoField.uint16("Innovusion", "latency_10us_max2", base.DEC, nil, nil, "")
    field_latency_10us_max3 = ProtoField.uint16("Innovusion", "latency_10us_max3", base.DEC, nil, nil, "")
    field_latency_10us_max4 = ProtoField.uint16("Innovusion", "latency_10us_max4", base.DEC, nil, nil, "")
    field_latency_10us_max5 = ProtoField.uint16("Innovusion", "latency_10us_max5", base.DEC, nil, nil, "")
    field_big_latency_frame = ProtoField.uint32("Innovusion", "big_latency_frame", base.DEC, nil, nil, "")
    field_bad_frame = ProtoField.uint32("Innovusion", "bad_frame", base.DEC, nil, nil, "")
    field_big_gap_frame = ProtoField.uint32("Innovusion", "big_gap_frame", base.DEC, nil, nil, "")
    field_small_gap_frame = ProtoField.uint32("Innovusion", "small_gap_frame", base.DEC, nil, nil, "")
    field_cpu_percentage = ProtoField.uint16("Innovusion", "cpu_percentage", base.DEC, nil, nil, "")
    field_mem_percentage = ProtoField.uint16("Innovusion", "mem_percentage", base.DEC, nil, nil, "")
    field_status_counter_motor0 = ProtoField.uint16("Innovusion", "motor0", base.DEC, nil, nil, "")
    field_status_counter_motor1 = ProtoField.uint16("Innovusion", "motor1", base.DEC, nil, nil, "")
    field_status_counter_motor2 = ProtoField.uint16("Innovusion", "motor2", base.DEC, nil, nil, "")
    field_status_counter_motor3 = ProtoField.uint16("Innovusion", "motor3", base.DEC, nil, nil, "")
    field_status_counter_motor4 = ProtoField.uint16("Innovusion", "motor4", base.DEC, nil, nil, "")
    field_status_counter_galvo0 = ProtoField.uint16("Innovusion", "galvo0", base.DEC, nil, nil, "")
    field_status_counter_galvo1 = ProtoField.uint16("Innovusion", "galvo1", base.DEC, nil, nil, "")
    field_status_counter_galvo2 = ProtoField.uint16("Innovusion", "galvo2", base.DEC, nil, nil, "")
    field_status_counter_galvo3 = ProtoField.uint16("Innovusion", "galvo3", base.DEC, nil, nil, "")
    field_status_counter_galvo4 = ProtoField.uint16("Innovusion", "galvo4", base.DEC, nil, nil, "")
    field_netstat_rx_speed_kBps = ProtoField.uint16("Innovusion", "netstat_rx_speed_kBps", base.DEC, nil, nil, "")
    field_netstat_tx_speed_kBps = ProtoField.uint16("Innovusion", "netstat_tx_speed_kBps", base.DEC, nil, nil, "")
    field_netstat_rx_drop = ProtoField.uint16("Innovusion", "netstat_rx_drop", base.DEC, nil, nil, "")
    field_netstat_tx_drop = ProtoField.uint16("Innovusion", "netstat_tx_drop", base.DEC, nil, nil, "")
    field_netstat_rx_err = ProtoField.uint16("Innovusion", "netstat_rx_err", base.DEC, nil, nil, "")
    field_netstat_tx_err = ProtoField.uint16("Innovusion", "netstat_tx_err", base.DEC, nil, nil, "")
    field_sys_cpu_percentage0 = ProtoField.uint16("Innovusion", "sys_cpu_percentage0", base.DEC, nil, nil, "")
    field_sys_cpu_percentage1 = ProtoField.uint16("Innovusion", "sys_cpu_percentage1", base.DEC, nil, nil, "")
    field_sys_cpu_percentage2 = ProtoField.uint16("Innovusion", "sys_cpu_percentage2", base.DEC, nil, nil, "")
    field_sys_cpu_percentage3 = ProtoField.uint16("Innovusion", "sys_cpu_percentage3", base.DEC, nil, nil, "")
    field_status_lifelong_uptime = ProtoField.uint32("Innovusion", "lifelong_uptime", base.DEC, nil, nil, "")

    --320
	function InnoStatusCounters_dissector(buf,pkt,parent)
        local h = parent:add("InnoStatusCounters")

        h:add_le(field_point_data_packet_sent,buf(0,8))
        h:add_le(field_point_sent,buf(8,8))
        h:add_le(field_message_packet_sent,buf(16,8))
        h:add_le(field_raw_data_read,buf(24,8))
        h:add_le(field_total_frame,buf(32,8))
        h:add_le(field_total_polygon_rotation,buf(40,8))
        h:add_le(field_total_polygon_facet,buf(48,8))
        h:add_le(field_power_up_time_in_second,buf(56,4))
        h:add_le(field_process_up_time_in_second,buf(60,4))
        h:add_le(field_lose_ptp_sync,buf(64,4))
        h:add_le(field_bad_data0,buf(68,4))
        h:add_le(field_bad_data1,buf(72,4))
        h:add_le(field_bad_data2,buf(76,4))
        h:add_le(field_bad_data3,buf(80,4))
        h:add_le(field_data_drop0,buf(84,4))
        h:add_le(field_data_drop1,buf(88,4))
        h:add_le(field_data_drop2,buf(92,4))
        h:add_le(field_data_drop3,buf(96,4))
        h:add_le(field_data_drop4,buf(100,4))
        h:add_le(field_data_drop5,buf(104,4))
        h:add_le(field_data_drop6,buf(108,4))
        h:add_le(field_data_drop7,buf(112,4))
        h:add_le(field_in_signals0,buf(116,4))
        h:add_le(field_in_signals1,buf(120,4))
        h:add_le(field_in_signals2,buf(124,4))
        h:add_le(field_in_signals3,buf(128,4))
        h:add_le(field_in_signals4,buf(132,4))
        h:add_le(field_in_signals5,buf(136,4))
        h:add_le(field_in_signals6,buf(140,4))
        h:add_le(field_in_signals7,buf(144,4))
        h:add_le(field_latency_10us_average0,buf(148,2))
        h:add_le(field_latency_10us_average1,buf(150,2))
        h:add_le(field_latency_10us_average2,buf(152,2))
        h:add_le(field_latency_10us_average3,buf(154,2))
        h:add_le(field_latency_10us_average4,buf(156,2))
        h:add_le(field_latency_10us_average5,buf(158,2))
        h:add_le(field_latency_10us_variation0,buf(160,2))
        h:add_le(field_latency_10us_variation1,buf(162,2))
        h:add_le(field_latency_10us_variation2,buf(164,2))
        h:add_le(field_latency_10us_variation3,buf(166,2))
        h:add_le(field_latency_10us_variation4,buf(168,2))
        h:add_le(field_latency_10us_variation5,buf(170,2))
        h:add_le(field_latency_10us_max0,buf(172,2))
        h:add_le(field_latency_10us_max1,buf(174,2))
        h:add_le(field_latency_10us_max2,buf(176,2))
        h:add_le(field_latency_10us_max3,buf(178,2))
        h:add_le(field_latency_10us_max4,buf(180,2))
        h:add_le(field_latency_10us_max5,buf(182,2))
        h:add_le(field_big_latency_frame,buf(184,4))
        h:add_le(field_bad_frame,buf(188,4))
        h:add_le(field_big_gap_frame,buf(192,4))
        h:add_le(field_small_gap_frame,buf(196,4))
        h:add_le(field_cpu_percentage,buf(200,2))
        h:add_le(field_mem_percentage,buf(202,2))
        h:add_le(field_status_counter_motor0,buf(204,2))
        h:add_le(field_status_counter_motor1,buf(206,2))
        h:add_le(field_status_counter_motor2,buf(208,2))
        h:add_le(field_status_counter_motor3,buf(210,2))
        h:add_le(field_status_counter_motor4,buf(212,2))
        h:add_le(field_status_counter_galvo0,buf(214,2))
        h:add_le(field_status_counter_galvo1,buf(216,2))
        h:add_le(field_status_counter_galvo2,buf(218,2))
        h:add_le(field_status_counter_galvo3,buf(220,2))
        h:add_le(field_status_counter_galvo4,buf(222,2))
        h:add_le(field_netstat_rx_speed_kBps,buf(224,2))
        h:add_le(field_netstat_tx_speed_kBps,buf(226,2))
        h:add_le(field_netstat_rx_drop,buf(228,2))
        h:add_le(field_netstat_tx_drop,buf(230,2))
        h:add_le(field_netstat_rx_err,buf(232,2))
        h:add_le(field_netstat_tx_err,buf(234,2))
        h:add_le(field_sys_cpu_percentage0,buf(236,2))
        h:add_le(field_sys_cpu_percentage1,buf(238,2))
        h:add_le(field_sys_cpu_percentage2,buf(240,2))
        h:add_le(field_sys_cpu_percentage3,buf(242,2))
        h:add_le(field_status_lifelong_uptime,buf(244,4))

    end
    -- InnoStatusSensorReadings
    field_temperature_fpga_10th_c = ProtoField.uint16("Innovusion", "temperature_fpga_10th_c", base.DEC, nil, nil, "")
    field_temperature_laser_10th_c = ProtoField.uint16("Innovusion", "temperature_laser_10th_c", base.DEC, nil, nil, "")
    field_temperature_adc_10th_c = ProtoField.uint16("Innovusion", "temperature_adc_10th_c", base.DEC, nil, nil, "")
    field_temperature_board_10th_c = ProtoField.uint16("Innovusion", "temperature_board_10th_c", base.DEC, nil, nil, "")
    field_temperature_det_10th_c0 = ProtoField.uint16("Innovusion", "temperature_det_10th_c0", base.DEC, nil, nil, "")
    field_temperature_det_10th_c1 = ProtoField.uint16("Innovusion", "temperature_det_10th_c1", base.DEC, nil, nil, "")
    field_temperature_det_10th_c2 = ProtoField.uint16("Innovusion", "temperature_det_10th_c2", base.DEC, nil, nil, "")
    field_temperature_det_10th_c3 = ProtoField.uint16("Innovusion", "temperature_det_10th_c3", base.DEC, nil, nil, "")
    field_temperature_other_10th_c0 = ProtoField.uint16("Innovusion", "temperature_other_10th_c0", base.DEC, nil, nil, "")
    field_temperature_other_10th_c1 = ProtoField.uint16("Innovusion", "temperature_other_10th_c1", base.DEC, nil, nil, "")
    field_temperature_other_10th_c2 = ProtoField.uint16("Innovusion", "temperature_other_10th_c2", base.DEC, nil, nil, "")
    field_heater_current_ma = ProtoField.uint16("Innovusion", "heater_current_ma", base.DEC, nil, nil, "")
    field_motor_rpm_1000th = ProtoField.uint32("Innovusion", "motor_rpm_1000th", base.DEC, nil, nil, "")
    field_galvo_fpm_1000th = ProtoField.uint32("Innovusion", "galvo_fpm_1000th", base.DEC, nil, nil, "")
    field_motor_rotation_total = ProtoField.uint64("Innovusion", "motor_rotation_total", base.DEC, nil, nil, "")
    field_galvo_round_total = ProtoField.uint64("Innovusion", "galvo_round_total", base.DEC, nil, nil, "")
    field_moisture_index0 = ProtoField.uint16("Innovusion", "moisture_index0", base.DEC, nil, nil, "")
    field_moisture_index1 = ProtoField.uint16("Innovusion", "moisture_index1", base.DEC, nil, nil, "")
    field_window_blockage_index0 = ProtoField.uint16("Innovusion", "window_blockage_index0", base.DEC, nil, nil, "")
    field_window_blockage_index1 = ProtoField.uint16("Innovusion", "window_blockage_index1", base.DEC, nil, nil, "")
    field_sensor_read_motor0 = ProtoField.uint16("Innovusion", "motor0", base.DEC, nil, nil, "")
    field_sensor_read_motor1 = ProtoField.uint16("Innovusion", "motor1", base.DEC, nil, nil, "")
    field_sensor_read_motor2 = ProtoField.uint16("Innovusion", "motor2", base.DEC, nil, nil, "")
    field_sensor_read_motor3 = ProtoField.uint16("Innovusion", "motor3", base.DEC, nil, nil, "")
    field_sensor_read_motor4 = ProtoField.uint16("Innovusion", "motor4", base.DEC, nil, nil, "")
    field_sensor_read_motor5 = ProtoField.uint16("Innovusion", "motor5", base.DEC, nil, nil, "")
    field_sensor_read_galvo0 = ProtoField.uint16("Innovusion", "galvo0", base.DEC, nil, nil, "")
    field_sensor_read_galvo1 = ProtoField.uint16("Innovusion", "galvo1", base.DEC, nil, nil, "")
    field_sensor_read_galvo2 = ProtoField.uint16("Innovusion", "galvo2", base.DEC, nil, nil, "")
    field_sensor_read_galvo3 = ProtoField.uint16("Innovusion", "galvo3", base.DEC, nil, nil, "")
    field_sensor_read_galvo4 = ProtoField.uint16("Innovusion", "galvo4", base.DEC, nil, nil, "")
    field_sensor_read_galvo5 = ProtoField.uint16("Innovusion", "galvo5", base.DEC, nil, nil, "")
    field_laser0 = ProtoField.uint16("Innovusion", "laser0", base.DEC, nil, nil, "")
    field_laser1 = ProtoField.uint16("Innovusion", "laser1", base.DEC, nil, nil, "")
    field_laser2 = ProtoField.uint16("Innovusion", "laser2", base.DEC, nil, nil, "")
    field_laser3 = ProtoField.uint16("Innovusion", "laser3", base.DEC, nil, nil, "")
    field_laser4 = ProtoField.uint16("Innovusion", "laser4", base.DEC, nil, nil, "")
    field_laser5 = ProtoField.uint16("Innovusion", "laser5", base.DEC, nil, nil, "")
    field_galvo_status_client = ProtoField.uint16("Innovusion", "galvo_status_client", base.DEC, nil, nil, "")
    field_galvo_offset_angle_client = ProtoField.uint16("Innovusion", "galvo_offset_angle_client", base.DEC, nil, nil, "")

    field_motor_dc_bus_voltage = ProtoField.uint32("Innovusion", "motor_dc_bus_voltage", base.DEC, nil, nil, "")
    field_motor_speed_control_err = ProtoField.uint16("Innovusion", "motor_speed_control_err", base.DEC, nil, nil, "")
    field_galvo_position_control_err = ProtoField.uint16("Innovusion", "galvo_position_control_err", base.DEC, nil, nil, "")
    field_unit_current = ProtoField.uint16("Innovusion", "unit_current", base.DEC, nil, nil, "")
    field_apd_bias_feedback0 = ProtoField.uint16("Innovusion", "apd_bias_feedback0", base.DEC, nil, nil, "")
    field_apd_bias_feedback1 = ProtoField.uint16("Innovusion", "apd_bias_feedback1", base.DEC, nil, nil, "")
    field_apd_bias_feedback2 = ProtoField.uint16("Innovusion", "apd_bias_feedback2", base.DEC, nil, nil, "")
    field_apd_bias_feedback3 = ProtoField.uint16("Innovusion", "apd_bias_feedback3", base.DEC, nil, nil, "")

		field_accel_x = ProtoField.uint16("Innovusion", "accel_x", base.DEC, nil, nil, "")
		field_accel_y = ProtoField.uint16("Innovusion", "accel_y", base.DEC, nil, nil, "")
		field_accel_z = ProtoField.uint16("Innovusion", "accel_z", base.DEC, nil, nil, "")
		field_gyro_x = ProtoField.uint16("Innovusion", "gyro_x", base.DEC, nil, nil, "")
		field_gyro_y = ProtoField.uint16("Innovusion", "gyro_x", base.DEC, nil, nil, "")
		field_gyro_z = ProtoField.uint16("Innovusion", "gyro_z", base.DEC, nil, nil, "")
		field_accel_unit_x = ProtoField.uint16("Innovusion", "accel_unit_x", base.DEC, nil, nil, "")
		field_accel_unit_y = ProtoField.uint16("Innovusion", "accel_unit_y", base.DEC, nil, nil, "")
		field_accel_unit_z = ProtoField.uint16("Innovusion", "accel_unit_z", base.DEC, nil, nil, "")
		field_gyro_unit_x = ProtoField.uint16("Innovusion", "gyro_unit_x", base.DEC, nil, nil, "")
		field_gyro_unit_y = ProtoField.uint16("Innovusion", "gyro_unit_y", base.DEC, nil, nil, "")
		field_gyro_unit_z = ProtoField.uint16("Innovusion", "yro_unit_z", base.DEC, nil, nil, "")
		field_gyro_temp = ProtoField.uint16("Innovusion", "gyro_temp", base.DEC, nil, nil, "")


    --falconigk  falcon2.1  robinw:192
    function InnoStatusSensorReading_dissector(buf,pkt,parent)
        local h = parent:add("InnoStatusSensorReading")

        h:add_le(field_temperature_fpga_10th_c,buf(0,2))
        h:add_le(field_temperature_laser_10th_c,buf(2,2))
        h:add_le(field_temperature_adc_10th_c,buf(4,2))
        h:add_le(field_temperature_board_10th_c,buf(6,2))
        h:add_le(field_temperature_det_10th_c0,buf(8,2))
        h:add_le(field_temperature_det_10th_c1,buf(10,2))
        h:add_le(field_temperature_det_10th_c2,buf(12,2))
        h:add_le(field_temperature_det_10th_c3,buf(14,2))
        h:add_le(field_temperature_other_10th_c0,buf(16,2))
        h:add_le(field_temperature_other_10th_c1,buf(18,2))
        h:add_le(field_temperature_other_10th_c2,buf(20,2))
        h:add_le(field_heater_current_ma,buf(22,2))
        h:add_le(field_motor_rpm_1000th,buf(24,4))
        h:add_le(field_galvo_fpm_1000th,buf(28,4))
        h:add_le(field_motor_rotation_total,buf(32,8))
        h:add_le(field_galvo_round_total,buf(40,8))
        h:add_le(field_moisture_index0,buf(48,2))
        h:add_le(field_moisture_index1,buf(50,2))
        h:add_le(field_window_blockage_index0,buf(52,2))
        h:add_le(field_window_blockage_index1,buf(54,2))
        h:add_le(field_sensor_read_motor0,buf(56,2))
        h:add_le(field_sensor_read_motor1,buf(58,2))
        h:add_le(field_sensor_read_motor2,buf(60,2))
        h:add_le(field_sensor_read_motor3,buf(62,2))
        h:add_le(field_sensor_read_motor4,buf(64,2))
        h:add_le(field_sensor_read_motor5,buf(66,2))
        h:add_le(field_sensor_read_galvo0,buf(68,2))
        h:add_le(field_sensor_read_galvo1,buf(70,2))
        h:add_le(field_sensor_read_galvo2,buf(72,2))
        h:add_le(field_sensor_read_galvo3,buf(74,2))
        h:add_le(field_sensor_read_galvo4,buf(76,2))
        h:add_le(field_sensor_read_galvo5,buf(78,2))
        h:add_le(field_laser0,buf(80,2))
        h:add_le(field_laser1,buf(82,2))
        h:add_le(field_laser2,buf(84,2))
        h:add_le(field_laser3,buf(86,2))
        h:add_le(field_laser4,buf(88,2))
        h:add_le(field_laser5,buf(90,2))
        h:add_le(field_galvo_status_client,buf(92,2))
        h:add_le(field_galvo_offset_angle_client,buf(94,2))

        h:add_le(field_motor_dc_bus_voltage, buf(96,4))
        h:add_le(field_motor_speed_control_err, buf(100,2))
        h:add_le(field_galvo_position_control_err, buf(102,2))
        h:add_le(field_unit_current, buf(104,2))
        h:add_le(field_apd_bias_feedback0, buf(106,2))
        h:add_le(field_apd_bias_feedback1, buf(108,2))
        h:add_le(field_apd_bias_feedback2, buf(110,2))
        h:add_le(field_apd_bias_feedback3, buf(112,2))

		h:add_le(field_accel_x, buf(114,2))
		h:add_le(field_accel_y, buf(116,2))
		h:add_le(field_accel_z, buf(118,2))
		h:add_le(field_gyro_x, buf(120,2))
		h:add_le(field_gyro_y, buf(122,2))
		h:add_le(field_gyro_z, buf(124,2))
		h:add_le(field_accel_unit_x, buf(126,4))
		h:add_le(field_accel_unit_y, buf(130,4))
		h:add_le(field_accel_unit_z, buf(134,4))
		h:add_le(field_gyro_unit_x, buf(138,4))
		h:add_le(field_gyro_unit_y, buf(142,4))
		h:add_le(field_gyro_unit_z, buf(146,4))
		h:add_le(field_gyro_temp, buf(150,2))
    end

	--[[ ---------------------------------------------------------------------
            struct InnoStatusPacket
            {
                InnoCommonHeader common;

                uint64_t idx;  /* global index of all InnoStatusPacket */

                uint8_t status_packet_interval_ms;  /* status packet send interval in ms    */
                uint8_t pre_lidar_mode;             /* previous InnoLidarMode               */
                uint16_t in_transition_mode_ms;  /* time (ms), LiDAR in the transition mode */

                char sn[16];                /* lidar serial number */
                uint16_t fault_version;

				uint16_t ref_count_enough_ts_ms;
                uint16_t ref_intensity[kInnoChannelNumber];
				uint8_t hw_num[3];
				uint8_t reserved;

                (
				--uint16_t reserve;
                --uint32_t reserved[3];
				)

                InnoStatusInFaults in_faults;
                InnoStatusExFaults ex_faults;
                InnoStatusCounters counters;
                InnoStatusSensorReadings sensor_readings;
            };
    --]] ---------------------------------------------------------------------
    local field_status_index = ProtoField.uint64("Innovusion", "idx", base.DEC, nil, nil, "status index, start from 0")

    local field_status_interval_ms = ProtoField.uint8("Innovusion", "status_packet_interval_ms", base.DEC, nil, nil, "status packet send interval in ms")
    local field_status_pre_lidar_mode = ProtoField.uint8("Innovusion", "pre_lidar_mode", base.DEC, nil, nil, "previous InnoLidarMode")
    local field_status_in_transition_mode_ms = ProtoField.uint16("Innovusion", "in_transition_mode_ms", base.DEC, nil, nil, "time (ms), LiDAR in the transition mode")
    -- InnoStatusExFaults
	local field_status_ex_fault = ProtoField.uint64("Innovusion", "ex_faults", base.HEX, nil, nil, "")

	local field_status_sn = ProtoField.string("Innovusion", "sn", base.ASCII, nil, nil, "lidar serial number")
	local field_status_fault_version = ProtoField.uint16("Innovusion", "fault_version", base.DEC, nil, nil, "")


	local field_status_ref_count_enough_ts_ms = ProtoField.uint16("Innovusion", "ref_count_enough_ts_ms", base.DEC, nil, nil, "")
	local field_status_ref_intensity0 = ProtoField.uint16("Innovusion", "ref_intensity0", base.DEC, nil, nil, "")
	local field_status_ref_intensity1 = ProtoField.uint16("Innovusion", "ref_intensity1", base.DEC, nil, nil, "")
	local field_status_ref_intensity2 = ProtoField.uint16("Innovusion", "ref_intensity2", base.DEC, nil, nil, "")
	local field_status_ref_intensity3 = ProtoField.uint16("Innovusion", "ref_intensity3", base.DEC, nil, nil, "")
	local field_status_hw_num0 = ProtoField.uint8("Innovusion", "hw_num0", base.DEC, nil, nil, "")
	local field_status_hw_num1 = ProtoField.uint8("Innovusion", "hw_num1", base.DEC, nil, nil, "")
	local field_status_hw_num2 = ProtoField.uint8("Innovusion", "hw_num2", base.DEC, nil, nil, "")
	local field_status_reserved = ProtoField.uint8("Innovusion", "reserved", base.DEC, nil, nil, "")



    --
    -- size:
    --
    function InnoStatusPacket_dissector(buf,pkt,parent)

        -- header
        InnoCommonHeader_dissector(buf(0,26), pkt, parent)

        -- other
        parent:add_le(field_status_index,buf(26,8))
        parent:add_le(field_status_interval_ms,buf(34,1))
        parent:add_le(field_status_pre_lidar_mode,buf(35,1))
        parent:add_le(field_status_in_transition_mode_ms,buf(36,2))
        parent:add_le(field_status_sn,buf(38,16))
        parent:add_le(field_status_fault_version,buf(54,2))


		parent:add_le(field_status_ref_count_enough_ts_ms, buf(56, 2))
		parent:add_le(field_status_ref_intensity0, buf(58, 2))
		parent:add_le(field_status_ref_intensity1, buf(60, 2))
		parent:add_le(field_status_ref_intensity2, buf(62, 2))
		parent:add_le(field_status_ref_intensity3, buf(64, 2))
		parent:add_le(field_status_hw_num0, buf(66, 1))
		parent:add_le(field_status_hw_num1, buf(67, 1))
		parent:add_le(field_status_hw_num2, buf(68, 1))
		parent:add_le(field_status_reserved, buf(69, 1))

			-- in fault
		InnoStatusInFaults_dissector(buf(70,12), pkt, parent)

			-- out fault:no use
		parent:add_le(field_status_ex_fault,buf(82,4))

			-- counters
		InnoStatusCounters_dissector(buf(86,320), pkt, parent)

			-- sensor_readings
		InnoStatusSensorReading_dissector(buf(406,192), pkt, parent)

    end





    -----------------------------------------------------------
    -- fields
    -----------------------------------------------------------
    p_sdkproto = Proto("innovusion", "Innovusion")

    p_sdkproto.fields = {
        ---- version
        field_magic_number, field_major_version, field_minor_version, field_fw_sequence,

        ---- header
        field_checksum, field_size, field_source_id, field_timestamp_sync_type, field_header_reserved,
        field_ts_start_us, field_lidar_mode, field_lidar_status,

        ---- data packet
        field_frame_index, field_frame_sub_idx, field_frame_sub_seq,
        field_item_type, field_item_number, field_item_size, field_topic,

        -- flags
        field_scanner_direction, field_use_reflectance, field_multi_return_mode, field_confidence_level,
        field_is_last_sub_frame, field_is_last_sequence, field_has_tail, field_reserved_flag,
		field_frame_sync_locked, field_is_first_sub_frame, field_last_four_channel,
		field_reserved_flag1,

		-- roi
        field_roi_h_angle, field_roi_v_angle,

        ---- status packet
        field_status_index, field_status_interval_ms, field_status_pre_lidar_mode, field_status_in_transition_mode_ms,
        field_status_sn, field_status_fault_version, field_status_ref_count_enough_ts_ms, field_status_ref_intensity0, field_status_ref_intensity1,
		field_status_hw_num0, field_status_hw_num1, field_status_hw_num2,
		field_status_ref_intensity2, field_status_ref_intensity3, field_status_reserved, field_status_ex_fault,

        field_infault_0, field_infault_1, field_infault_2, field_infault_3, field_infault_4, field_infault_5, field_infault_6, field_infault_7,
        field_infault_8, field_infault_9, field_infault_10, field_infault_11, field_infault_12, field_infault_13, field_infault_14, field_infault_15,
        field_infault_16, field_infault_17, field_infault_18, field_infault_19, field_infault_20, field_infault_21, field_infault_22, field_infault_23,
        field_infault_24, field_infault_25, field_infault_26, field_infault_27, field_infault_28, field_infault_29, field_infault_30, field_infault_31,
        field_infault_32, field_infault_33, field_infault_34, field_infault_35, field_infault_36, field_infault_37, field_infault_38, field_infault_39,
        field_infault_40, field_infault_41, field_infault_42, field_infault_43, field_infault_44, field_infault_45, field_infault_46, field_infault_47,
        field_infault_48, field_infault_49, field_infault_50, field_infault_51, field_infault_52, field_infault_53, field_infault_54, field_infault_55,
        field_infault_56, field_infault_57, field_infault_58, field_infault_59, field_infault_60, field_infault_61, field_infault_62, field_infault_63,
        field_extended_faults,

        ---- status counters
        field_point_data_packet_sent, field_point_sent, field_message_packet_sent, field_raw_data_read, field_total_frame, field_total_polygon_rotation,
        field_total_polygon_facet, field_power_up_time_in_second, field_process_up_time_in_second, field_lose_ptp_sync, field_bad_data0, field_bad_data1,
        field_bad_data2, field_bad_data3, field_data_drop0, field_data_drop1, field_data_drop2, field_data_drop3, field_data_drop4, field_data_drop5,
        field_data_drop6, field_data_drop7, field_in_signals0, field_in_signals1, field_in_signals2, field_in_signals3, field_in_signals4, field_in_signals5,
        field_in_signals6, field_in_signals7, field_latency_10us_average0, field_latency_10us_average1, field_latency_10us_average2, field_latency_10us_average3,
        field_latency_10us_average4, field_latency_10us_average5, field_latency_10us_variation0, field_latency_10us_variation1, field_latency_10us_variation2,
        field_latency_10us_variation3, field_latency_10us_variation4, field_latency_10us_variation5, field_latency_10us_max0, field_latency_10us_max1,
        field_latency_10us_max2, field_latency_10us_max3, field_latency_10us_max4, field_latency_10us_max5, field_big_latency_frame, field_bad_frame,
        field_big_gap_frame, field_small_gap_frame, field_cpu_percentage, field_mem_percentage, field_status_counter_motor0, field_status_counter_motor1, field_status_counter_motor2,
        field_status_counter_motor3, field_status_counter_motor4,
        field_status_counter_galvo0, field_status_counter_galvo1, field_status_counter_galvo2, field_status_counter_galvo3, field_status_counter_galvo4, field_netstat_rx_speed_kBps,
        field_netstat_tx_speed_kBps, field_netstat_rx_drop,
        field_netstat_tx_drop, field_netstat_rx_err, field_netstat_tx_err, field_sys_cpu_percentage0, field_sys_cpu_percentage1, field_sys_cpu_percentage2,
        field_sys_cpu_percentage3, field_status_lifelong_uptime,

        ---- status sensor reading
        field_temperature_fpga_10th_c, field_temperature_laser_10th_c, field_temperature_adc_10th_c, field_temperature_board_10th_c, field_temperature_det_10th_c0, field_temperature_det_10th_c1,
        field_temperature_det_10th_c2, field_temperature_det_10th_c3, field_temperature_other_10th_c0, field_temperature_other_10th_c1, field_temperature_other_10th_c2, field_heater_current_ma,
        field_motor_rpm_1000th, field_galvo_fpm_1000th, field_motor_rotation_total, field_galvo_round_total, field_moisture_index0, field_moisture_index1, field_window_blockage_index0,
        field_window_blockage_index1, field_sensor_read_motor0, field_sensor_read_motor1, field_sensor_read_motor2, field_sensor_read_motor3, field_sensor_read_motor4, field_sensor_read_motor5,
        field_sensor_read_galvo0, field_sensor_read_galvo1, field_sensor_read_galvo2, field_sensor_read_galvo3, field_sensor_read_galvo4, field_sensor_read_galvo5,
        field_laser0, field_laser1, field_laser2, field_laser3, field_laser4, field_laser5, field_galvo_status_client,
        field_galvo_offset_angle_client, field_motor_dc_bus_voltage, field_motor_speed_control_err,field_galvo_position_control_err, field_unit_current, field_apd_bias_feedback0,
        field_apd_bias_feedback1, field_apd_bias_feedback2, field_apd_bias_feedback3,
		field_accel_x,field_accel_y,field_accel_z,field_gyro_x,field_gyro_y,field_gyro_z,field_accel_unit_x,
        field_accel_unit_y,field_accel_unit_z,field_gyro_unit_x,field_gyro_unit_y,field_gyro_unit_z,field_gyro_temp,
        ---- InnoMessage
        field_msg_size, field_msg_src, field_msg_id, field_msg_level, field_msg_code, field_msg_reserved, field_msg_content,

        ---- InnoXyzPoint
        field_xyz_x, field_xyz_y, field_xyz_z, field_xyz_radius, field_xyz_ts_10us,
        field_xyz_scan_id, field_xyz_in_roi, field_xyz_facet, field_xyz_reserved_flags, field_xyz_multi_return,
        field_xyz_is_2nd_return, field_xyz_scan_idx,field_xyz_refl, field_xyz_type, field_xyz_elongation, field_xyz_channel, field_xyz_ring_id,

		field_xyz_scan_id1, field_xyz_in_roi1, field_xyz_facet1, field_xyz_multi_return1, field_xyz_scan_idx1, field_xyz_type1, field_xyz_refl1, field_xyz_elongation1, field_xyz_channel1,field_xyz_is_2nd_return1,field_xyz_reserved_flags1,
        ---- InnoChannelPoint
        field_channel_radius, field_channel_refl, field_channel_is_2nd_return, field_channel_type, field_channel_elongation, field_channel_intensity, field_channel_reserved,
		field_channel_refl1, field_channel_radius1, field_channel_elongation1, field_channel_is_2nd_return1, field_channel_type1, field_channel_firing,

        field_channel_refl_rw_generic,
        field_channel_radius_rw_generic,

        ---- InnoBlockHeader
        field_bh_h_angle,
        field_bh_v_angle,

        field_bh_ts_10us,
        field_bh_scan_idx,

        field_bh_scan_id,

        field_bh_h_angle_diff_1,
        field_bh_h_angle_diff_2,
        field_bh_h_angle_diff_3,

        field_bh_v_angle_diff_1,
        field_bh_v_angle_diff_2,
        field_bh_v_angle_diff_3,

		field_bh_h_angle_diff_11,
        field_bh_h_angle_diff_21,
        field_bh_h_angle_diff_31,

        field_bh_v_angle_diff_11,
        field_bh_v_angle_diff_21,
        field_bh_v_angle_diff_31,

        field_bh_p_angle_diff_rw_generic,
        field_bh_g_angle_diff_rw_generic,
        field_channel_is_2nd_return_rw_generic,
        field_channel_firing_rw_generic,

        field_bh_in_roi,
        field_bh_facet,
        field_bh_reserved_flags
    }


    --
    -- InnoDataPacket
    --
    function InnoDataPacket_dissector(buf,pkt,parent)
        -- header
        InnoCommonHeader_dissector(buf(0,26),pkt,parent)

        -- other
        parent:add_le(field_frame_index,buf(26,8))
        parent:add_le(field_frame_sub_idx,buf(34,2))
        parent:add_le(field_frame_sub_seq,buf(36,2))

        parent:add_le(field_item_type,buf(38,4))
        parent:add_le(field_item_number,buf(38,4))
        parent:add_le(field_item_size,buf(42,2))
        parent:add_le(field_topic,buf(44,4))

        parent:add_le(field_scanner_direction,buf(48,2))
        parent:add_le(field_use_reflectance,buf(48,2))
        parent:add_le(field_multi_return_mode,buf(48,2))
        parent:add_le(field_confidence_level,buf(48,2))
        parent:add_le(field_is_last_sub_frame,buf(48,2))
        parent:add_le(field_is_last_sequence,buf(48,2))
        parent:add_le(field_has_tail,buf(48,2))
		if is_falcon_igk == true
		then
			parent:add_le(field_reserved_flag,buf(48,2))

			parent:add_le(field_roi_h_angle,buf(50,2))
			parent:add_le(field_roi_v_angle,buf(52,2))
		else
			parent:add_le(field_frame_sync_locked,buf(48,2))
			parent:add_le(field_is_first_sub_frame,buf(48,2))
			parent:add_le(field_last_four_channel, buf(48,2))
			parent:add_le(field_reserved_flag1,buf(48,2))
			parent:add_le(field_roi_h_angle,buf(50,2))
			parent:add_le(field_roi_v_angle,buf(52,2))
		end
        -- body
        local item_type = buf(38,1):le_uint()
        local itme_size = buf(42,2):le_uint()

		if is_falcon_igk == true
		then
            data_half_len = 54
            innoblock1_len = 33
            innoblock2_len = 49
            innoxyzpoint_len = 26
        elseif is_robinw_generic then
            data_half_len = 70
            innocoblock1_len = 42
            innocoblock2_len = 74
			innoxyzpoint_len = 26
		else
			data_half_len = 70
			innoblock1_len = 50
			innoblock2_len = 82
			innoxyzpoint_len = 30
		end


		local itme_num = (buf:len() - data_half_len) / itme_size

        if (item_type == 1 or item_type == 7 or item_type == 9) then
            -- innovusion:sphere
            if (itme_size == innoblock1_len ) then
                for i = 0, itme_num - 1, 1 do
                    InnoBlock1_dissector(buf(data_half_len + i*innoblock1_len, innoblock1_len),pkt,parent, i)
                end
            elseif (itme_size == innoblock2_len ) then
                for i = 0, itme_num - 1, 1 do
                    InnoBlock2_dissector(buf(data_half_len + i*innoblock2_len, innoblock2_len),pkt,parent, i)
                end
            else
                --
            end
        elseif item_type == 13 then
            -- innovusion:robinw_generic sphere
            if itme_size == innocoblock1_len then
                for i = 0, itme_num - 1, 1 do
                    InnoBlock1_dissector(buf(data_half_len + i*innocoblock1_len, innocoblock1_len),pkt,parent, i)
                end
            elseif itme_size == innocoblock2_len then
                for i = 0, itme_num - 1, 1 do
                    InnoBlock2_dissector(buf(data_half_len + i*innocoblock2_len, innocoblock2_len),pkt,parent, i)
                end
            else
                --
            end
        elseif (item_type == 2) then
            -- innovusion:message
            InnoMessage_dissector(buf(data_half_len),pkt,parent)
        elseif (item_type == 3) then
            -- innovusion:message_log
            InnoMessage_dissector(buf(data_half_len),pkt,parent)
        elseif (item_type == 4 or item_type == 8 or item_type == 10) then
            -- innovusion:xyz_pc size
            if (itme_size == innoxyzpoint_len ) then
                for i = 0, itme_num - 1, 1 do
                    InnoXyzPoint_dissector(buf(data_half_len + i*innoxyzpoint_len, innoxyzpoint_len),pkt,parent)
                end
            end
        else
            --
        end
    end


--
    -- ScoreBoard
    function ScoreBoard_dissector(buf,pkt,root)
        local buf_len = buf:len();
        if buf_len < 40 then return false end

        -- local magic_number_sw = bit.bswap(magic_number)

        -- root
        local t = root:add(p_sdkproto,buf(0,buf_len))

		local major_version = buf(2,1):uint()
		if major_version > 1
		then
            local item_type = buf(38,1):le_uint()
            if (item_type == 13) then
                is_robinw_generic = true
            end
			is_falcon_igk = false
		else
			is_falcon_igk = true
		end

        local magic_number = buf(0,2):le_uint()
        if (magic_number == 0x176a) then
            local item_type = buf(38,1):uint()
            if (item_type == 1 or item_type == 7 or item_type == 9 or item_type == 13) then
                pkt.cols.protocol = "innovusion:sphere"
            elseif (item_type == 2) then
                pkt.cols.protocol = "innovusion:message"
            elseif (item_type == 3) then
                pkt.cols.protocol = "innovusion:message_log"
            elseif (item_type == 4 or item_type == 8 or item_type == 10) then
                pkt.cols.protocol = "innovusion:xyz"
            else
                pkt.cols.protocol = "innovusion:data"
            end
            InnoDataPacket_dissector(buf,pkt,t)
        elseif (magic_number == 0x186b) then
            pkt.cols.protocol = "innovusion:status"
            InnoStatusPacket_dissector(buf, pkt, t)
        else
            return false
        end

        return true
    end

    --
    -- dissector
    --
    data_dis = Dissector.get("data")

    function p_sdkproto.dissector(buf,pkt,root)
        if ScoreBoard_dissector(buf,pkt,root)
		then
        else
            data_dis:call(buf,pkt,root)
        end
    end
    if tcpopen == false then
        local udp_table = DissectorTable.get("udp.port")
        udp_table:add(8010,p_sdkproto)
        udp_table:add("8000-65535",p_sdkproto)
    else
        local tcp_table = DissectorTable.get("tcp.port")
        tcp_table:add(8010, p_sdkproto)
        tcp_table:add("8000-65535", p_sdkproto)
    end
end
