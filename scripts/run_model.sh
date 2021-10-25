cd ..
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)"/db"
layer="bert_layer2"

cd temp

timeloop-model $SHELL_FOLDER/arch/components/* arch.yaml \
    $SHELL_FOLDER/vgg16/$layer".yaml" dataflow.yaml