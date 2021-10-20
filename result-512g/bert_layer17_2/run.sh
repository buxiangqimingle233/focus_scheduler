ARCH_FOLDER="/home/wangzhao/timeloop-dev/focus/db/arch"
PROB_FOLDER="/home/wangzhao/timeloop-dev/focus/db/bert"

timeloop-model $ARCH_FOLDER/components/* modified_arch.yaml \
    dump_mapping.yaml $PROB_FOLDER/bert_layer17.yaml