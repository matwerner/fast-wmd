####################     PARAMETERS     ####################

DATASET_NAMES=$1
PARTITIONS=$2
DISTANCE_METHODS=$3
LIMITERS=$4
VERBOSE=$5

###############     RUN     ####################

export SCRIPT_FILEPATH=$(readlink -f "$0")
export SCRIPT_DIRPATH=$(dirname "$SCRIPT_FILEPATH")
export DISTANCE_METHODS
export LIMITERS
export VERBOSE

for name in ${DATASET_NAMES};
do
	export DATASET_PATH="../dataset/kusner/${name}"

	for i in ${PARTITIONS};
	do
		export TRAIN_DATASET="${name}-train-${i}.txt"
		export TEST_DATASET="${name}-test-${i}.txt"
		export EMBEDDING="${name}-embeddings.txt"

		sh ${SCRIPT_DIRPATH}/experiments.sh
	done
done
