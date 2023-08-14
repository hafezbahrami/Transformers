from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence


def generate_addition(a, b):
    """
    a               b               carry       ===>        text
   ----           -----           --------                --------
    4               35                0                    "4+35;"

    step %10
    4               5                 0                  "4+35;450"

    carry=(a+b+carry)%10=9            9
                                                          "4+35;4509"
    resetting a, b ==> a=a//10 && b=b//10, carry= a + b + (carry%10)
    0               3                 0                 "4+35;4509030"

    go to step 1 till a=b=carry = 0
                                                      "4+35;4509030000=39"

    """
    text = "%d+%d;" % (a, b)
    carry = 0
    s = a + b
    while True:
        text += "%d%d%d" % (a % 10, b % 10, carry)
        if a == 0 and b == 0 and carry == 0:
            break
        text += "%d" % ((a + b + carry) % 10)
        carry = (a % 10 + b % 10 + carry) // 10
        a = a // 10
        b = b // 10
    text += "=%d" % s
    return text


def generate_multiplication_single(a, b):
    """
    a = 99256
    b = 7

    **Product Step**
    s1 = [6X7, 5X7, 2X7, 9X7, 9X7] = [42, 35, 14, 63, 63]
    text = '99256*7;6*7;=42;5*7;=35;2*7;=14;9*7;=63;9*7;=63;'

    **Addition Step**
    s1[0]       42 ==========================>           2
    s1[1]     +35                                  text(4+35)="4+35;45090303000=39"
        ------------------------------------------------
               39  ==========================>          92
    s1[2]    +14                                   text(3+14)="3+14;34070101000=17"
        ------------------------------------------------
              17   ==========================>         792
    s1[3]   +63                                    text(1+63)="1+63;13040606000=64"
        ------------------------------------------------
             64    ==========================>        4792
    s1[4]  +63                                     text(6+63)="6+63;63090606000=69"
        ------------------------------------------------
            69     ==========================>      694792
                                                   text="=694792"
    """
    if a < 10:
        return "%d*%d;=%d" % (a, b, a * b)
    product = a * b
    text = "%d*%d;" % (a, b)
    prods = []
    while a > 0:
        prods.append((a % 10) * b)
        text += generate_multiplication_single(a % 10, b)
        text += ";"
        a = a // 10
    # Do additions
    s = prods[0]
    for i in range(1, len(prods)):
        s = s // 10
        text += generate_addition(s, prods[i])
        text += ";"
        s += prods[i]
    text += "=%d" % product
    return text


def generate_multiplication(a, b):
    if b < 10:
        return generate_multiplication_single(a, b)
    product = a * b
    text = "%d*%d;" % (a, b)
    # Single digit multiplications
    prods = []
    while b > 0:
        prods.append(a * (b % 10))
        text += generate_multiplication_single(a, b % 10)
        text += ";"
        b = b // 10
    # Do additions
    s = prods[0]
    for i in range(1, len(prods)):
        s = s // 10
        text += generate_addition(s, prods[i])
        text += ";"
        s += prods[i]
    text += "=%d" % product
    return text


def get_multiplication_answer(s):
    try:
        eq = s.rindex('=')
        ed = s.rindex('$')
        return int(s[eq + 1:ed])
    except:
        return None


class ExpressionDataset(Dataset):
    def __init__(self, count, begin=0, n_digit=5):
        super(ExpressionDataset, self).__init__()
        self.count = count
        self.begin = begin
        self.n_digit = n_digit

    def __getitem__(self, i):
        assert i < self.count and i >= 0, "Out of bounds"
        rng = np.random.RandomState(i + self.begin)
        x = rng.randint(low=10**(self.n_digit-1), high=10**self.n_digit-1, size=2, dtype=np.int64)
        return generate_multiplication(x[0], x[1]) + '$'

    def __len__(self):
        return self.count


class TestDataset(Dataset):
    def __init__(self, count, begin=0, n_digit=5):
        super(ExpressionDataset, self).__init__()
        self.count = count
        self.begin = begin
        self.n_digit = n_digit

    def __getitem__(self, i):
        assert i < self.count and i >= 0, "Out of bounds"
        rng = np.random.RandomState(i + self.begin)
        x = rng.randint(low=10**(self.n_digit-1), high=10**self.n_digit-1, size=2, dtype=np.int64)
        return ((x[0], x[1]), x[0] * x[1])

    def __len__(self):
        return self.count


def data_loader(count=1000, begin=0, n_digit=5, batch_size=32, shuffle=True, valid_data_set=True):
    """
    The output loader is a list of len XXX. Each item in the list has Batch# of strings for multiplications-in-string.
    """
    dataset = ExpressionDataset(count=count, begin=begin, n_digit=n_digit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader