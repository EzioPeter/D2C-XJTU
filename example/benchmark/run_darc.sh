#!/usr/bin/env bash

cd ./example/benchmark || { echo "Unable to enter the workspace!"; exit 1; }
EXECUTION_COUNT=5
GPU_DEVICE=4

for ENV_U in 'HalfCheetah'; do
    ENV="${ENV_U}-v2"
    for DATA_TYPE in 'medium_replay-v2' 'medium_expert-v2' 'medium-v2'; do
        DATA="${ENV_U,,}_${DATA_TYPE}"
        for DYNAMICS_TYPE in 'gravity' 'friction' 'thigh_size' 'thigh_range' 'torso_length'; do
            case $DYNAMICS_TYPE in
                gravity)
                    DEGREES=(2.0)
                    ;;
                friction)
                    DEGREES=(0.3)
                    ;;
                thigh_size|thigh_range)
                    DEGREES=(2.0 0.5)
                    ;;
                torso_length)
                    DEGREES=(4.0)
                    ;;
            esac

            for DEGREE in "${DEGREES[@]}"; do
                echo "Starting experiments for: env=$ENV, data=$DATA, dynamics=$DYNAMICS_TYPE, degree=$DEGREE"
                
                for ((i=1; i<=$EXECUTION_COUNT; i++)); do
                    echo "Execution $i/$EXECUTION_COUNT"
                    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python demo_darc.py \
                        --env_name="$ENV" \
                        --data_name="$DATA" \
                        --unreal_dynamics="$DYNAMICS_TYPE" \
                        --variety_degree="$DEGREE"
                    
                    if [ $? -eq 0 ]; then
                        echo "Execution $i succeeded"
                    else
                        echo "Execution $i failed"
                    fi
                    
                    sleep 2
                done
            done
        done
    done
done

echo "All executions completed"
