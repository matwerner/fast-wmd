EXE="../build/fast_wmd"

DATASET_NAME=$1
DISTANCE_METHODS=$2
LIMITERS=$3
NUM_CLUSTERS=$4
MAX_ITER=$5

if [ -z "$LIMITERS" ];            then LIMITERS=-1;                fi
if [ -z "$NUM_CLUSTERS" ];        then NUM_CLUSTERS=-1;            fi
if [ -z "$MAX_ITER" ];            then MAX_ITER=-1;                fi

FILEPATH_PREFIX="../dataset/triplets/${DATASET_NAME}/${DATASET_NAME}"
EXE_BASIC_PARAMS="triplets --trip ${FILEPATH_PREFIX}-triplets.txt --docs ${FILEPATH_PREFIX}-papers.txt --emb ${FILEPATH_PREFIX}-embeddings.txt"

for DISTANCE_METHOD in ${DISTANCE_METHODS}
do
	for LIMITER in ${LIMITERS}
	do
		${EXE} ${EXE_BASIC_PARAMS} --num_clusters ${NUM_CLUSTERS} --max_iter ${MAX_ITER} --func ${DISTANCE_METHOD} --r ${LIMITER}
	done
done

