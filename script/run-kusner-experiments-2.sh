####################     PARAMETERS     ####################

DATASET_NAMES=$1
DISTANCE_METHODS=$2
LIMITERS=$3
VERBOSE=$4

###############     RUN     ####################

export SCRIPT_FILEPATH=$(readlink -f "$0")
export SCRIPT_DIRPATH=$(dirname "$SCRIPT_FILEPATH")
export DISTANCE_METHODS
export LIMITERS
export VERBOSE

for name in ${DATASET_NAMES};
do
    export DATASET_PATH="../dataset/kusner/${name}"

    export TRAIN_DATASET="${name}-train.txt"
    export TEST_DATASET="${name}-test.txt"
    export EMBEDDING="${name}-embeddings.txt"

    sh $SCRIPT_DIRPATH/experiments.sh
done
