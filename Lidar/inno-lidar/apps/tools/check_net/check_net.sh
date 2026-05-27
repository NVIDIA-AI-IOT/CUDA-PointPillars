#!/usr/bin/env sh

#
# sudo apt install iperf3
# sudo apt install curl
# sudo apt install wget
#

SCRIPT=$(basename "$0")
SCRIPTBASE="${SCRIPT%.*}"
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# region initialization functions ----------------------------------------------
minInit() {
    IPERF_TIME=15            # s
    GET_PCD_TIME=300        # s
    GOOD_TCP_BITRATE=33     # Mb/s
    OK_TCP_BITRATE=30       # Mb/s
    GOOD_UDP_BITRATE=18   # Mb/s
    OK_UDP_BITRATE=15      # Mb/s
    GOOD_UDP_JITTER=0.75    # ms
    OK_UDP_JITTER=1.25      # ms
    REBOOT_TIME=15          # s

    LIDAR_IP=172.168.1.10
    APPS_DIR=$(dirname "$SCRIPTPATH")
    GET_PCD=$APPS_DIR/get_pcd/get_pcd
    LIDAR_UTIL="$APPS_DIR/lidar_util/innovusion_lidar_util"
    echo $LIDAR_UTIL
    setRunDT
    LOG_DIR="$SCRIPTPATH/logs"
    if [ ! -d "$LOG_DIR" ]
    then
        mkdir -p "$LOG_DIR"
    else
        rm -Rf "$LOG_DIR"/*.log
    fi
    SCRIPT_LOG="$LOG_DIR/$SCRIPTBASE.$RUN_DT.log"
    touch "$SCRIPT_LOG"
    DO_GET_PCD_TEST=1
    DO_IPERF_TEST=1
    IGNORE_ZERO_XFER=0
    KEEP_INNO_PC=0
    VERBOSE=0
    ISSUES=0
    ISSUE_REPORT=""
    ATTENTION="< < < < < < < < < < < < < < < < < < < <"
    HRULE="--------------------------------------------------"
    HRULE2="=================================================="
}

scriptUsage() {
    printf "%s takes optional parameters:\n" "$SCRIPT"
    printf "\t-n|--lidar-ip:     IP address of attached Falcon (default %s)\n" "$LIDAR_IP"
    printf "\t-k|--keep-inno-pc: don't delete test .inno_pc files (default false)\n"
    printf "\t--skip-get-pcd:    don't run get_pcd tests (default false)\n"
    printf "\t--skip-iperf:      don't run iperf tests (default false)\n"
    printf "\t-v|--verbose:      writes more information to stdout (default false)\n"
}

parseArgs() {
    while [ "$#" -gt 0 ]
    do
        case $1 in
            -h|--help)
                scriptUsage
                exit 0
                ;;
            -k|--keep-inno-pc)
                KEEP_INNO_PC=1
                ;;
            --skip-get-pcd)
                DO_GET_PCD_TEST=0
                ;;
            --skip-iperf)
                DO_IPERF_TEST=0
                ;;
            -v|--verbose)
                VERBOSE=1
                ;;
            -n|--lidar-ip)
                LIDAR_IP=$2
                shift
                ;;
            *)
                echo "Default case (*): argument $1 ignored"
        esac
        shift
    done
    RUN_LIDAR_UTIL="$LIDAR_UTIL $LIDAR_IP"
}
# endregion initialization functions -------------------------------------------

# region boilerplate helper functions ------------------------------------------
contains() {
    string="$1"
    substring="$2"
    if test "${string#*$substring}" != "$string"
    then
        return 0    # $substring is in $string
    else
        return 1    # $substring is not in $string
    fi
}

getURL() {
    GET_URL=$1
    DESC=$2
    if [ $MISSING_CURL -gt 0 ]
    then
        verbosePrint "$SCRIPT_LOG" "$DESC: using wget to request $GET_URL"
        "$WGET" -o /tmp/wget.log -O /tmp/"$LIDAR_IP".reboot.html "$GET_URL"
    else
        verbosePrint "$SCRIPT_LOG" "$DESC: using curl to request $GET_URL"
        "$CURL" "$GET_URL" >> "$SCRIPT_LOG" 2>&1
    fi
}

pleaseWait() {
    verbosePrint "$SCRIPT_LOG" "Waiting for long-running processing"
    CHECK_ON_PID=0
    WAIT_TIME=$1
    if [ $# -ge 2 ]
    then
        WAIT_MSG=$2
    else
        WAIT_MSG="Please wait...."
    fi
    if [ $# -eq 3 ]
    then
        CHECK_ON_PID=$3
    fi
    WAIT_COUNTER=1
    while [ "$WAIT_COUNTER" -le "$WAIT_TIME" ]
    do
        printf "\r%s -- %d" "$WAIT_MSG" "$WAIT_COUNTER"
        WAIT_COUNTER=$((WAIT_COUNTER+1))
        sleep 1

        if [ "$CHECK_ON_PID" -ne "0" ]
        then
            ps cax | grep "$CHECK_ON_PID" > /dev/null
            if [ $? -eq 1 ]
            then
                MSG="$CHECK_ON_PID finished ($WAIT_COUNTER < $WAIT_TIME)"
                verbosePrint "$SCRIPT_LOG" "Stop waiting early: $MSG"
                WAIT_COUNTER=$((WAIT_COUNTER+WAIT_TIME))
            else
                if [ "$((WAIT_COUNTER % 10))" -eq 0 ]
                then
                    MSG="$CHECK_ON_PID is still alive ($WAIT_COUNTER < $WAIT_TIME)"
                    verbosePrint "$SCRIPT_LOG" "$MSG"
                fi
            fi
        fi
    done
    printf "\n"
    verbosePrint "$SCRIPT_LOG" "Long-running processing COMPLETE"
}

saveArray() {
    for i do printf %s\\n "$i" | sed "s/'/'\\\\''/g;1s/^/'/;\$s/\$/' \\\\/" ; done
    echo " "
}

setRunDT() {
    RUN_DT=$(date +%Y%m%d-%H%M%S)
}

stopProcessByPID() {
    CHECK_PID=$1
    verbosePrint "$SCRIPT_LOG" "Checking on $CHECK_PID"
    ps cax | grep "$CHECK_PID" > /dev/null
    if [ $? -eq 0 ]
    then
        teePrint "$SCRIPT_LOG" "$CHECK_PID still alive -- need to kill it"
        pkill -P "$CHECK_PID" > /dev/null
        kill "$CHECK_PID" > /dev/null
    fi
}

teePrint() {
    setRunDT
    teePrintBase "$@"
}

teePrintBase() {
    TEE_LOGFILE="$1"
    shift
    MSG="$*"
    if [ -e "$TEE_LOGFILE" ]
    then
        echo "$RUN_DT | $MSG" >> "$TEE_LOGFILE"
    fi
    echo "$RUN_DT | $MSG"
}

toInt() {
    FLOAT_VAL=$1
    INT_RESULT=$(printf '%.*f\n' 0 "$FLOAT_VAL")
}

verbosePrint() {
    # Always log the message, but only send to STDOUT if verbosity ON
    if [ $VERBOSE -eq 1 ]
    then
        teePrint "$@"
    else
        TEE_LOGFILE="$1"
        shift
        MSG="$*"
        if [ -e "$TEE_LOGFILE" ]
        then
            setRunDT
            echo "$RUN_DT | $MSG" >> "$TEE_LOGFILE"
        fi
    fi
}
# endregion boilerplate helper functions ---------------------------------------

# region lidar functions -------------------------------------------------------
rebootSensor() {
    REBOOT_URL="http://$LIDAR_IP:8010/command/?set_reboot=1"
    getURL "$REBOOT_URL" "REBOOT"
    REBOOT_MSG="Rebooting the sensor to apply new configuration / environment"
    pleaseWait $REBOOT_TIME "$REBOOT_MSG"
}

startInternalPCS() {
    teePrint "$SCRIPT_LOG" "Starting internal PCS on lidar at $LIDAR_IP"
    START_PCS_URL="http://$LIDAR_IP:8675/v1/pcs/enable?toggle=on"
    getURL "$START_PCS_URL" "START PCS"
}

stopInternalPCS() {
    teePrint "$SCRIPT_LOG" "Stopping internal PCS on lidar at $LIDAR_IP"
    STOP_PCS_URL="http://$LIDAR_IP:8675/v1/pcs/enable?toggle=off"
    getURL "$STOP_PCS_URL" "STOP PCS"
}
# endregion lidar functions ----------------------------------------------------

# region iperf testing functions -----------------------------------------------
addIssue() {
    ISSUE_MSG=$1
    if ! contains "$ISSUE_REPORT" "$ISSUE_MSG"; then
        ISSUE_REPORT="\t$ISSUE_MSG\n$ISSUE_REPORT"
    fi
}

checkLineUnits() {
    LOG_LINE=$1
    LINE_UNITS=$(echo "$LOG_LINE" | awk '{print $8}')
    if [ "$LINE_UNITS" != "Mbits/sec" ]
    then
        DATA_XFER=$(echo "$LOG_LINE" | awk '{print $7}')
        if [ $IGNORE_ZERO_XFER -eq 0 ]
        then
            verbosePrint "$SCRIPT_LOG" "Zero bytes transferred will be flagged!!!!"
            if [ "$DATA_XFER" != "0.00" ]
            then
                TOO_SLOW=1
                teePrint "$SCRIPT_LOG" "##$LINE_UNITS## too slow (! Mbits/sec) $ATTENTION"
                addIssue "slow connection"
            fi
        fi
    else
        TOO_SLOW=0
    fi
}

splitLog() {
    INPUT_LOG=$1
    SPLIT_AT_LINE=$2
    OUTPUT_LOG1=$3
    OUTPUT_LOG2=$4
    LINE_TOTAL=$(wc -l < $INPUT_LOG)
    LOG2_LINES=$(( LINE_TOTAL - SPLIT_AT_LINE ))
    head -n "$SPLIT_AT_LINE" "$INPUT_LOG" > "$OUTPUT_LOG1"
    tail -n "$LOG2_LINES" "$INPUT_LOG" > "$OUTPUT_LOG2"
}

goodOKLess() {
    TEST_VAL=$1
    UNITS=$2
    OK_VAL=$3
    GOOD_VAL=$4
    ISSUE_MSG=$5
    if [ $IGNORE_ZERO_XFER -eq 1 ]
    then
        toInt $TEST_VAL
        if [ $INT_RESULT -eq 0 ]
        then
            verbosePrint "$SCRIPT_LOG" "Ignoring measured $TEST_VAL data transfer"
            return
        fi
    fi

    if [ $MISSING_BCALC -eq 0 ]
    then
        VAL_OK=$(echo "$TEST_VAL < $OK_VAL" | bc)
        VAL_GOOD=$(echo "$TEST_VAL < $GOOD_VAL" | bc)
        if [ "$VAL_OK" -eq 1 ]
        then
            ISSUES=1
            teePrint "$SCRIPT_LOG" "$TEST_VAL $UNITS is too small $ATTENTION"
            addIssue "$ISSUE_MSG"
        elif [ "$VAL_GOOD" -eq 1 ]
        then
            teePrint "$SCRIPT_LOG" "$TEST_VAL $UNITS may be borderline"
        else
            verbosePrint "$SCRIPT_LOG" "$TEST_VAL $UNITS is GOOD!!!!"
        fi
    else
        toInt $TEST_VAL
        TEST_VAL=$INT_RESULT
        if [ $TEST_VAL -lt $OK_VAL ]
        then
            ISSUES=1
            teePrint "$SCRIPT_LOG" "$TEST_VAL $UNITS is too small $ATTENTION"
            addIssue "$ISSUE_MSG"
        elif [ $TEST_VAL -lt $GOOD_VAL ]
        then
            teePrint "$SCRIPT_LOG" "$TEST_VAL $UNITS may be borderline"
        else
            verbosePrint "$SCRIPT_LOG" "$TEST_VAL $UNITS is GOOD!!!!"
        fi
    fi
}

goodOKGreater() {
    TEST_VAL=$1
    UNITS=$2
    OK_VAL=$3
    GOOD_VAL=$4
    ISSUE_MSG=$5

    if [ $MISSING_BCALC -eq 0 ]
    then
        VAL_OK=$(echo "$TEST_VAL > $OK_VAL" | bc)
        VAL_GOOD=$(echo "$TEST_VAL > $GOOD_VAL" | bc)
        if [ "$VAL_OK" -eq 1 ]
        then
            # jitter measurement does not appear to be reliable, so don't flag
            # high jitter as something to investigate right now.  Reconsider,
            # if newer iperf3 fixes this....
            # ISSUES=1
            teePrint "$SCRIPT_LOG" "$TEST_VAL $UNITS is high"
            # addIssue "$ISSUE_MSG"
        elif [ "$VAL_GOOD" -eq 1 ]
        then
            teePrint "$SCRIPT_LOG" "$TEST_VAL $UNITS may be borderline"
        else
            verbosePrint "$SCRIPT_LOG" "$TEST_VAL $UNITS is GOOD!!!!"
        fi
    else
        toInt $TEST_VAL
        TEST_VAL=$INT_RESULT
        if [ $TEST_VAL -gt $OK_VAL ]
        then
            # jitter measurement does not appear to be reliable, so don't flag
            # high jitter as something to investigate right now.  Reconsider,
            # if newer iperf3 fixes this....
            # ISSUES=1
            teePrint "$SCRIPT_LOG" "$TEST_VAL $UNITS is high"
            # addIssue "$ISSUE_MSG"
        elif [ $TEST_VAL -gt $GOOD_VAL ]
        then
            teePrint "$SCRIPT_LOG" "$TEST_VAL $UNITS may be borderline"
        else
            verbosePrint "$SCRIPT_LOG" "$TEST_VAL $UNITS is GOOD!!!!"
        fi
    fi
}

checkBitrateTCP() {
    goodOKLess $1 "Mbits/sec" $OK_TCP_BITRATE $GOOD_TCP_BITRATE "slow connection"
}

parseIperfTCP() {
    IPERF_LOG="$1"
    BASE_LOG=$(basename "$IPERF_LOG")
    CLIENT_LOG="/tmp/client.$BASE_LOG"
    SERVER_LOG="/tmp/server.$BASE_LOG"
    CLIENT_LNES=$(awk '! NF { print NR; exit }' "$IPERF_LOG")
    splitLog $IPERF_LOG $CLIENT_LNES $CLIENT_LOG $SERVER_LOG

    IGNORE_ZERO_XFER=0
    verbosePrint "$SCRIPT_LOG" "Checking CLIENT TCP bitrate in $BASE_LOG"
    verbosePrint "$SCRIPT_LOG" "IGNORE_ZERO_XFER: $IGNORE_ZERO_XFER"
    parseIperfTCPLog "$CLIENT_LOG"

    IGNORE_ZERO_XFER=1
    verbosePrint "$SCRIPT_LOG" "Checking SERVER TCP bitrate in $BASE_LOG"
    verbosePrint "$SCRIPT_LOG" "IGNORE_ZERO_XFER: $IGNORE_ZERO_XFER"
    parseIperfTCPLog "$SERVER_LOG"
}

parseIperfTCPLog() {
    IPERF_LOG="$1"
    RECV_LINE=$(grep receiver "$IPERF_LOG")
    verbosePrint "$SCRIPT_LOG" "RECEIVER:\n$RECV_LINE"
    checkLineUnits "$RECV_LINE"
    if [ $TOO_SLOW -eq 0 ]
    then
        RECV_BITRATE=$(echo "$RECV_LINE" | awk '{print $7}')
        checkBitrateTCP "$RECV_BITRATE"
    fi

    SEND_LINE=$(grep sender "$IPERF_LOG")
    verbosePrint "$SCRIPT_LOG" "SENDER:\n$SEND_LINE"
    checkLineUnits "$SEND_LINE"
    if [ $TOO_SLOW -eq 0 ]
    then
        SEND_BITRATE=$(echo "$SEND_LINE" | awk '{print $7}')
        checkBitrateTCP "$SEND_BITRATE"
    fi
}

checkBitrateUDP() {
    goodOKLess $1 "Mbits/sec" $OK_UDP_BITRATE $GOOD_UDP_BITRATE "slow connection"
}

checkJitter() {
    goodOKGreater $1 "ms" $OK_UDP_JITTER $GOOD_UDP_JITTER "too much jitter"
}

parseIperfUDP() {
    IPERF_LOG="$1"
    BASE_LOG=$(basename "$IPERF_LOG")
    CLIENT_LOG="/tmp/client.$BASE_LOG"
    SERVER_LOG="/tmp/server.$BASE_LOG"
    CLIENT_LNES=$(awk '! NF { print NR; exit }' "$IPERF_LOG")
    splitLog $IPERF_LOG $CLIENT_LNES $CLIENT_LOG $SERVER_LOG
    CHECK_MSG="UDP bitrate and jitter in"
    verbosePrint "$SCRIPT_LOG" "Checking CLIENT $CHECK_MSG $BASE_LOG"
    parseIperfUDPLog "$CLIENT_LOG"

    verbosePrint "$SCRIPT_LOG" "Checking SERVER $CHECK_MSG $BASE_LOG"
    parseIperfUDPLog "$SERVER_LOG"
}

parseIperfUDPLog() {
    IPERF_LOG="$1"
    RECV_LINE=$(grep receiver "$IPERF_LOG")
    if [ "$RECV_LINE" = "" ]
    then
        SUMMARY_LINE=$(grep "0.00-$IPERF_TIME" "$IPERF_LOG")
        if [ "$SUMMARY_LINE" = "" ]
        then
            ISSUES=1
            DISP_LOG=$(basename $IPERF_LOG)
            teePrint "$SCRIPT_LOG" "$DISP_LOG missing bitrate & jitter values $ATTENTION"
            addIssue "iperf output missing"
        else
            checkLineUnits "$SUMMARY_LINE"
            if [ $TOO_SLOW -eq 0 ]
            then
                SUMMARY_BITRATE=$(echo "$SUMMARY_LINE" | cut -c 37-43 | sed 's/ //g')
                checkBitrateUDP "$SUMMARY_BITRATE"
            fi
            SUMMARY_JITTER=$(echo "$SUMMARY_LINE" | awk '{print $9}')
            verbosePrint "$SCRIPT_LOG" "Jitter check:\n$SUMMARY_LINE\n\t->$SUMMARY_JITTER"
            checkJitter "$SUMMARY_JITTER"
        fi
    else
        checkLineUnits "$RECV_LINE"
        if [ $TOO_SLOW -eq 0 ]
        then
            RECV_BITRATE=$(echo "$RECV_LINE" | cut -c 37-43 | sed 's/ //g')
            checkBitrateUDP "$RECV_BITRATE"
        fi
        RECV_JITTER=$(echo "$RECV_LINE" | awk '{print $9}')
        verbosePrint "$SCRIPT_LOG" "Jitter check:\n$RECV_LINE\n\t->$RECV_JITTER"
        checkJitter "$RECV_JITTER"

        SEND_LINE=$(grep sender "$IPERF_LOG")
        checkLineUnits "$SEND_LINE"
        if [ $TOO_SLOW -eq 0 ]
        then
            SEND_BITRATE=$(echo "$SEND_LINE" | cut -c 37-43 | sed 's/ //g')
            checkBitrateUDP "$SEND_BITRATE"
        fi
        SEND_JITTER=$(echo "$SEND_LINE" | awk '{print $9}')
        verbosePrint "$SCRIPT_LOG" "Jitter check:\n$SEND_LINE\n\t->$SEND_JITTER"
        checkJitter "$SEND_JITTER"
    fi
}

testIperf() {
    teePrint "$SCRIPT_LOG" $HRULE
    teePrint "$SCRIPT_LOG" "Running iperf test"
    stopInternalPCS
    rebootSensor

    IPERF_VER=$($IPERF -v)
    verbosePrint "$SCRIPT_LOG" "iperf is $IPERF\n$IPERF_VER"
    set -- "" "--reverse"
    IPERF_ARGS=$(saveArray "$@")
    if [ $MISSING_BCALC -eq 0 ]
    then
        set -- "" "--reverse" "--udp -b 20M" "--udp --reverse -b 20M"
        # set -- "--udp" "--udp --reverse"
        IPERF_ARGS=$(saveArray "$@")
    fi
    COUNT=0
    LAST_ARG=$#
    while [ $COUNT -lt $LAST_ARG ]
    do
        COUNT=$(( COUNT + 1 ))
        TEST_ARG=$1
        shift

        verbosePrint "$SCRIPT_LOG" $HRULE
        GET_SRV_OUT_ARG="--get-server-output"
        BASE_ARGS="-c $LIDAR_IP -t $IPERF_TIME $GET_SRV_OUT_ARG"
        IPERF_CMD="$IPERF $BASE_ARGS $TEST_ARG"
        teePrint "$SCRIPT_LOG" "$IPERF_CMD"
        FILENAME_ARGS=$(echo "$TEST_ARG" | sed 's/ //g')
        IPERF_LOG="$LOG_DIR"/iperf"$FILENAME_ARGS".log

        $IPERF_CMD > "$IPERF_LOG" 2>&1 &
        IPERF_PID=$!
        teePrint "$SCRIPT_LOG" "Running in background as $IPERF_PID"

        # provide slight buffer for iperf initialization & cleanup
        WAIT_FOR=$((IPERF_TIME+1))
        RUN_IPERF_MSG="collect about $IPERF_TIME sec of network data"
        pleaseWait $WAIT_FOR "$RUN_IPERF_MSG" $IPERF_PID

        # make sure get_pcd (and descendents) is (are) stopped
        stopProcessByPID $IPERF_PID

        cat "$IPERF_LOG" >> "$SCRIPT_LOG"

        # check for issues in $IPERF_LOG
        case "$TEST_ARG" in
            *udp*)
                parseIperfUDP "$IPERF_LOG"
                ;;
            *)
                parseIperfTCP "$IPERF_LOG"
                ;;
        esac
    done

    startInternalPCS
    rebootSensor
}
# endregion iperf testing functions --------------------------------------------

# region get_pcd testing functions ---------------------------------------------
getPCSENV() {
    SAVE_AS=$1
    $RUN_LIDAR_UTIL download_internal_file PCS_ENV "$SAVE_AS"
}

setPCSENV() {
    NEW_PCS_ENV=$1
    SET_PCS_ENV_ARGS="upload_internal_file PCS_ENV $NEW_PCS_ENV"
    SET_PCS_ENV_CMD="$RUN_LIDAR_UTIL $SET_PCS_ENV_ARGS"
    teePrint "$SCRIPT_LOG" "./innovusion_lidar_util $LIDAR_IP $SET_PCS_ENV_ARGS"
    $SET_PCS_ENV_CMD
}

parseGetPCDLog() {
    PCD_LOGFILE=$1
    FRAME_COUNT=$2
    verbosePrint "$SCRIPT_LOG" "Parsing get_pcd output logged in $PCD_LOGFILE"

    # do not want to see any message lines with '[ WARN]', '[ERROR]' or '[CRITI]'
    # but focus on bandwidth only....
    WARN_BWIDTH=$(grep bandwidth "$PCD_LOGFILE" | grep WARN)
    if [ "$WARN_BWIDTH" != "" ]
    then
        teePrint "$SCRIPT_LOG" "Bandwidth warning(s): $ATTENTION\n$WARN_BWIDTH"
        ISSUES=$((ISSUES+1))
        addIssue "get_pcd bandwidth warning(s)"
    fi

    # expect to see 'frame_counter = $FRAME_COUNT' at end
    EXPECTED_FRAMECOUNT=$(grep "frame_counter = $FRAME_COUNT" "$PCD_LOGFILE")
    if [ "$EXPECTED_FRAMECOUNT" = "" ]
    then
        teePrint "$SCRIPT_LOG" "Did not receive expected $FRAME_COUNT frames $ATTENTION"
        addIssue "get_pcd missing frame(s)"
        ISSUES=$((ISSUES+1))
    fi

    # also expect 'miss_frame_counter = 0' at end
    EXPECTED_MISSED=$(grep "miss_frame_counter = 0" "$PCD_LOGFILE")
    if [ "$EXPECTED_MISSED" = "" ]
    then
        teePrint "$SCRIPT_LOG" "Non-zero missed frames $ATTENTION"
        addIssue "get_pcd missing frame(s)"
        ISSUES=$((ISSUES+1))
    fi

    verbosePrint "$SCRIPT_LOG" $HRULE
}

doGetPCD() {
    MODE=$1
    GET_PCD_FRAMES=$((GET_PCD_TIME*10))
    case $MODE in
        BROADCAST)
            MODE_ARGS=""
            ;;
        UNICAST)
            MODE_ARGS="--lidar-udp-port 8010"
            ;;
        TCP)
            MODE_ARGS="--lidar-udp-port 0 --use-tcp"
            ;;
    esac
    GET_PCD_ARGS="--lidar-ip $LIDAR_IP --frame-number $GET_PCD_FRAMES $MODE_ARGS"
    GET_PCD_LOG="$LOG_DIR/get_pcd.$MODE.log"
    INNO_PC_FILE="$LOG_DIR/$MODE.inno_pc"
    SAVE_INNO_PC="-P $INNO_PC_FILE"
    GET_PCD_CMD="$GET_PCD $GET_PCD_ARGS $SAVE_INNO_PC"
    verbosePrint "$SCRIPT_LOG" $HRULE
    teePrint "$SCRIPT_LOG" "$GET_PCD_CMD > $GET_PCD_LOG 2>&1 &"
    $GET_PCD_CMD > "$GET_PCD_LOG" 2>&1 &
    GET_PCD_PID=$!
    teePrint "$SCRIPT_LOG" "Running in background as $GET_PCD_PID"

    # provide buffer for get_pcd to go through initialization and winding down
    WAIT_FOR=$((GET_PCD_TIME+15))
    RUN_PCD_MSG="collect about 5 minutes of data to ensure reasonably accurate results"
    pleaseWait $WAIT_FOR "$RUN_PCD_MSG" $GET_PCD_PID

    # make sure get_pcd (and descendents) is (are) stopped
    stopProcessByPID $GET_PCD_PID

    # make note of how big the inno_pc & log files are
    OUTPUT_FILES=$(ls -l "$LOG_DIR"/*"$MODE"*)
    verbosePrint "$SCRIPT_LOG" "\n$OUTPUT_FILES"

    # check stdout (& stderr) from get_pcd log
    parseGetPCDLog "$GET_PCD_LOG" "$GET_PCD_FRAMES"

    if [ $KEEP_INNO_PC -eq 0 ]
    then
        # cleanup -- .inno_pc files are big
        rm -Rf "$INNO_PC_FILE"
    fi
}

makePCSEnvs() {
    PCS_ENV_ORIG="$LIDAR_IP"_PCS_ENV_ORIG
    getPCSENV "$PCS_ENV_ORIG"

    # duplicate PCS_ENV -- set UDP broadcast
    UDP_BROADCAST="${LIDAR_IP%.*}.255"
    cp "$PCS_ENV_ORIG" PCS_ENV_BROADCAST
    sed -i "/UDP_IP=/c\UDP_IP=$UDP_BROADCAST" PCS_ENV_BROADCAST

    # duplicate PCS_ENV -- set UDP unicast
    cp "$PCS_ENV_ORIG" PCS_ENV_UNICAST
    sed -i "/UDP_IP=/c\# UDP_IP=$UDP_BROADCAST" PCS_ENV_UNICAST

    # UDP unicast also works for TCP
    cp PCS_ENV_UNICAST PCS_ENV_TCP
}

testGetPCD() {
    teePrint "$SCRIPT_LOG" $HRULE
    teePrint "$SCRIPT_LOG" "Running get_pcd test"
    makePCSEnvs

    for MODE in BROADCAST UNICAST TCP; do
        teePrint "$SCRIPT_LOG" $HRULE

        # upload PCS_ENV copy to sensor
        setPCSENV "PCS_ENV_$MODE"

        # reboot the sensor to use the new UDP mode
        rebootSensor

        # use get_pcd to record frames from live sensor
        doGetPCD $MODE
    done

    # restore original PCS_ENV to sensor & reboot
    setPCSENV "$PCS_ENV_ORIG"
    rebootSensor
}
# endregion get_pcd testing functions ------------------------------------------

alertPrereqs() {
    MISSING_UTIL=0
    teePrint "$SCRIPT_LOG" "Checking pre-requisites...."
    teePrint "$SCRIPT_LOG" $HRULE2

    # region iperf prereq
    # need iperf3 (or iperf)
    IPERF=$(which iperf3)
    MISSING_IPERF3=$?
    if [ $MISSING_IPERF3 -gt 0 ]
    then
        teePrint "$SCRIPT_LOG" $HRULE
        teePrint "$SCRIPT_LOG" "iperf3 not installed."
        IPERF=$(which iperf)
        MISSING_IPERF=$?
        if [ $MISSING_IPERF -gt 0 ]
        then
            MISSING_UTIL=$((MISSING_IPERF+MISSING_IPERF3))
            teePrint "$SCRIPT_LOG" "iperf also not installed."
            INSTALL_MSG="sudo apt-get update\n\tsudo apt-get install iperf3"
            teePrint "$SCRIPT_LOG" "Please install 'iperf3':\n\t$INSTALL_MSG"
        fi
    fi
    # endregion iperf prereq

    # region curl prereq
    # need curl (or wget)
    CURL=$(which curl)
    MISSING_CURL=$?
    if [ $MISSING_CURL -gt 0 ]
    then
        teePrint "$SCRIPT_LOG" $HRULE
        teePrint "$SCRIPT_LOG" "curl not installed."
        WGET=$(which wget)
        MISSING_WGET=$?
        if [ $MISSING_WGET -gt 0 ]
        then
            MISSING_UTIL=$((MISSING_UTIL+MISSING_CURL+MISSING_WGET))
            teePrint "$SCRIPT_LOG" "wget also not installed."
            INSTALL_MSG="sudo apt-get update\n\tsudo apt-get install curl"
            teePrint "$SCRIPT_LOG" "Please install 'curl':\n\t$INSTALL_MSG"
        fi
    fi
    # endregion curl prereq

    # region bc prereq
    # bc (basic calculator) is helpful, not critical
    BCALC=$(which bc)
    MISSING_BCALC=$?
    if [ $MISSING_BCALC -gt 0 ]
    then
        teePrint "$SCRIPT_LOG" $HRULE
        teePrint "$SCRIPT_LOG" "bc (basic calculator) not installed."
        teePrint "$SCRIPT_LOG" "NOTICE: iperf UDP data transfer check will be skipped."
        INSTALL_MSG="sudo apt-get update\n\tsudo apt-get install bc"
        teePrint "$SCRIPT_LOG" "Install 'bc' to check UDP data transfer:\n\t$INSTALL_MSG"
    fi
    # endregion curl prereq

    # region lidar_util prereq
    # need innovusion_lidar_util
    if [ ! -e "../lidar_util/innovusion_lidar_util" ]
    then
        teePrint "$SCRIPT_LOG" $HRULE
        teePrint "$SCRIPT_LOG" "innovusion_lidar_util is missing."
        MISSING_UTIL=$((MISSING_UTIL+1))
        teePrint "$SCRIPT_LOG" "Please run 'make' to recompile $LIDAR_UTIL"
    fi
    # endregion lidar_util prereq

    # region net connection / lidar power
    ping -c 1 -i 0.2 -W 1 "$LIDAR_IP" >> "$SCRIPT_LOG" 2>&1
    if [ $? -ne 0 ]
    then
        teePrint "$SCRIPT_LOG" $HRULE
        teePrint "$SCRIPT_LOG" "Lidar is offline; cannot connect to $LIDAR_IP"
        teePrint "$SCRIPT_LOG" "Please check sensor power & network connections."
        MISSING_UTIL=$((MISSING_UTIL+1))
    fi
    # endregion net connection / lidar power

    if [ $MISSING_UTIL -gt 0 ]
    then
        teePrint "$SCRIPT_LOG" $HRULE2
        teePrint "$SCRIPT_LOG" "Please address above and then re-run this script."
        exit $MISSING_UTIL
    fi
}

alertIssues() {
    teePrint "$SCRIPT_LOG" $HRULE2
    MSG1="Network performance issues were found:"
    MSG2="Please send contents of\n\t$LOG_DIR\nto Innovusion for analysis."
    teePrint "$SCRIPT_LOG" "$MSG1\n$ISSUE_REPORT\n$MSG2"
}

mainCheck() {
    minInit
    parseArgs "$@"
    alertPrereqs
    if [ $DO_IPERF_TEST -ne 0 ]; then
        testIperf
    fi
    if [ $DO_GET_PCD_TEST -ne 0 ]; then
        testGetPCD
    fi
    teePrint "$SCRIPT_LOG" $HRULE2
    teePrint "$SCRIPT_LOG" "Testing complete."
    if [ $ISSUES -gt 0 ]
    then
        alertIssues
        exit $ISSUES
    fi
}

mainCheck "$@"
