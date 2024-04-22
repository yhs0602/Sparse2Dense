FROM ghcr.io/yhs0602/craftground/release:1.5.9

WORKDIR /workspace
ADD old_examples/any_structure.py /workspace/any_structure.py
ADD requirements.txt /workspace/requirements.txt

RUN pip install -r requirements.txt
