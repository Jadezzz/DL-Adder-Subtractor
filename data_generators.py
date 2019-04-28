import numpy as np

def generate_addition_data(train_size, digits, ctable, chars):
    
    data_size = train_size + 2000 + 40000
    maxlen = digits + 1 + digits
    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    while len(questions) < data_size:
        f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, digits + 1))))
        a, b = f(), f()
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        q = '{}+{}'.format(a, b)
        query = q + ' ' * (maxlen - len(q))
        ans = str(a + b)
        ans += ' ' * (digits + 1 - len(ans))
        questions.append(query)
        expected.append(ans)

    print('Total addition questions:', len(questions))
    print('Example questions:')
    print(questions[:5], expected[:5])
    
    print()
    
    x = np.zeros((len(questions), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(expected), digits + 1, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, maxlen)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, digits + 1)

    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # train_test_split
    train_x = x[:-42000]
    train_y = y[:-42000]
    val_x = x[-42000:-40000]
    val_y = y[-42000:-40000]
    test_x = x[-40000:]
    test_y = y[-40000:]
    

    print('Training Data:')
    print(train_x.shape)
    print(train_y.shape)

    print('Validation Data:')
    print(val_x.shape)
    print(val_y.shape)

    print('Testing Data:')
    print(test_x.shape)
    print(test_y.shape)
    
    print()
    
    return train_x, train_y, val_x, val_y, test_x, test_y


def generate_subtraction_data(train_size, digits, ctable, chars):
    
    data_size = train_size + 2000 + 40000
    maxlen = digits + 1 + digits
    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    while len(questions) < data_size:
        f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, digits + 1))))
        a, b = f(), f()
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        big = max(a,b)
        small = min(a,b)
        q = '{}-{}'.format(big, small)
        query = q + ' ' * (maxlen - len(q))
        ans = str(big - small)
        ans += ' ' * (digits - len(ans))
        questions.append(query)
        expected.append(ans)

    print('Total addition questions:', len(questions))
    print('Example questions:')
    print(questions[:5], expected[:5])

    print()
    
    x = np.zeros((len(questions), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(expected), digits, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, maxlen)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, digits)

    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # train_test_split
    train_x = x[:-42000]
    train_y = y[:-42000]
    val_x = x[-42000:-40000]
    val_y = y[-42000:-40000]
    test_x = x[-40000:]
    test_y = y[-40000:]
    

    print('Training Data:')
    print(train_x.shape)
    print(train_y.shape)

    print('Validation Data:')
    print(val_x.shape)
    print(val_y.shape)

    print('Testing Data:')
    print(test_x.shape)
    print(test_y.shape)
    
    print()
    
    return train_x, train_y, val_x, val_y, test_x, test_y


def generate_add_sub_data(train_size, digits, ctable, chars):
    
    data_size = train_size + 2000 + 40000
    maxlen = digits + 1 + digits
    questions = []
    expected = []
    seen = set()
    count = 0
    print('Generating data...')
    while len(questions) < data_size:
        f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, digits + 1))))
        a, b = f(), f()
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        
        if count < data_size/2:
            q = '{}+{}'.format(a, b)
            query = q + ' ' * (maxlen - len(q))
            ans = str(a + b)
            ans += ' ' * (digits + 1 - len(ans))
        else:
            big = max(a,b)
            small = min(a,b)
            q = '{}-{}'.format(big, small)
            query = q + ' ' * (maxlen - len(q))
            ans = str(big - small)
            ans += ' ' * (digits + 1 - len(ans))
        questions.append(query)
        expected.append(ans)
        count += 1

    print('Total combined questions:', len(questions))
    print('Example questions:')
    print(questions[:5], expected[:5])
    print(questions[-5:], expected[-5:])
    
    print()

    x = np.zeros((len(questions), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(expected), digits + 1, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, maxlen)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, digits + 1)

    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # train_test_split
    train_x = x[:-42000]
    train_y = y[:-42000]
    val_x = x[-42000:-40000]
    val_y = y[-42000:-40000]
    test_x = x[-40000:]
    test_y = y[-40000:]
    

    print('Training Data:')
    print(train_x.shape)
    print(train_y.shape)

    print('Validation Data:')
    print(val_x.shape)
    print(val_y.shape)

    print('Testing Data:')
    print(test_x.shape)
    print(test_y.shape)
    
    print()
    
    return train_x, train_y, val_x, val_y, test_x, test_y