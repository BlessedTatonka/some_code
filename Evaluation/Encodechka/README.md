# encodechka
This is a fork from the [original encodechka](https://github.com/avidale/encodechka) repository.

There is a fixed sklearn argument ('log' -> 'log_loss') and useful script.

```python
git clone https://github.com/BlessedTatonka/encodechka
cd encodechka
pip install -r requirements.txt

# To run on NER tasks | ~2 mins.
python3 run_encodechka.py --model <model> --tasks NE1,NE2

# To run on all tasks (IC and ICX computation takes time) | ~20 mins.
python3 run_encodechka.py --model <model>
```
