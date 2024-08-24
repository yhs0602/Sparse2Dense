#!/bin/bash

# Start Xvfb
Xvfb :2 -screen 0 1024x768x24 +extension GLX -ac +extension RENDER &

# Set the DISPLAY environment variable
export DISPLAY=:2
export PYTHONPATH=.
export WANDB_MODE=offline
export WANDB_DIR='../results/'

# Configure X11 screen settings
xset s off
xset -dpms
xset q

# Default values for the parameters
TYPE=""
EXCLUDE_GOAL=0
EXTENDED_WALL=0
REWARD=""
GOAL=""
TRANSITION_TIMING=""

# Parse the arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --type=*) TYPE="${1#*=}" ;;
        --exclude_goal=*) EXCLUDE_GOAL="${1#*=}" ;;
        --extended_wall=*) EXTENDED_WALL="${1#*=}" ;;
        --reward=*) REWARD="${1#*=}" ;;
        --goal=*) GOAL="${1#*=}" ;;
        --transition-timing=*) TRANSITION_TIMING="${1#*=}" ;;
        --type) TYPE="$2"; shift ;;
        --exclude_goal) EXCLUDE_GOAL="$2"; shift ;;
        --extended_wall) EXTENDED_WALL=1; shift ;;  # store-true behavior
        --reward) REWARD="$2"; shift ;;
        --goal) GOAL="$2"; shift ;;
        --transition-timing) TRANSITION_TIMING="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Determine the command to run based on the type and reward
COMMAND=""

if [[ "$TYPE" == "room" ]]; then
    COMMAND="conda run --no-capture-output -n minecraft_maze python room/experiments/${REWARD}.py"
    if [[ "$REWARD" == "transition" || "$REWARD" == "d2s" ]]; then
        if [[ -z "$TRANSITION_TIMING" ]]; then
            echo "Error: --transition-timing must be specified for transition or d2s reward."
            exit 1
        fi
        COMMAND="$COMMAND --transition-timing ${TRANSITION_TIMING}"
    fi
    if [[ "$EXTENDED_WALL" -eq 1 ]]; then
        COMMAND="$COMMAND --extended"
    fi

elif [[ "$TYPE" == "cross_w2" ]]; then
    COMMAND="conda run --no-capture-output -n minecraft_maze python cross_w2/experiments/${REWARD}.py"
    if [[ -z "$GOAL" ]]; then
        echo "Error: --goal must be specified for cross_w2 experiments."
        exit 1
    fi
    COMMAND="$COMMAND --goal ${GOAL}"

    if [[ "$REWARD" == "transition" || "$REWARD" == "d2s" ]]; then
        if [[ -z "$TRANSITION_TIMING" ]]; then
            echo "Error: --transition-timing must be specified for transition or d2s reward."
            exit 1
        fi
        COMMAND="$COMMAND --transition-timing ${TRANSITION_TIMING}"
    fi

else
    echo "Invalid type specified: $TYPE"
    exit 1
fi

# Run the constructed command
$COMMAND --verbose
