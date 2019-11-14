EXE="../build/fast_wmd"

##########    DEFAULT VALUES    ##########

if [ -z "$LIMITERS" ];            then LIMITERS=-1;                fi
if [ -z "$VERBOSE" ];             then VERBOSE="false";            fi

##########    OUR ALGORITHMS     ##########

for DISTANCE_METHOD in ${DISTANCE_METHODS}
do
    for LIMITER in ${LIMITERS}
    do
        ${EXE} kusner --tr "${DATASET_PATH}/${TRAIN_DATASET}" --te "${DATASET_PATH}/${TEST_DATASET}" --emb "${DATASET_PATH}/${EMBEDDING}" --func $DISTANCE_METHOD --k 19 --r $LIMITER --verbose $VERBOSE
    done
done
