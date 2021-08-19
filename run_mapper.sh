SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)"/db"
LAYER="bert-large_layer10"

echo "working for $LAYER"

cd result-512g/$LAYER"_4"
timeloop-mapper $SHELL_FOLDER/arch/components/* modified_arch.yaml \
    $SHELL_FOLDER/constraints/* $SHELL_FOLDER/mapper/* $SHELL_FOLDER/bert-large/$LAYER".yaml"