REPO="$(basename $PWD)"
VENV="venv_$REPO"
if [ ! -d "$VENV" ]; then
    echo "Creating venv in $PWD/$VENV/"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    uv venv $VENV
    echo "Venv created in $PWD/$VENV/"
fi

export PYTHONPATH="$PYTHONPATH:$(pwd)"

source $VENV/bin/activate
echo "Venv activated"

uv pip install --upgrade pip
uv pip install -r requirements.txt
uv pip install ipykernel
echo "Pip requirements installed"

python3 -m ipykernel install --user --name="$VENV" --display-name="$REPO"
echo "Created ipykernel"
jupyter kernelspec list

pre-commit install
echo "Process finished"