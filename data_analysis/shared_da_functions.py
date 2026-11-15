import copy

def normalise_cons_time_series(sets):
    normalised = {}
    for key, df in sets.items():
        ## This only works when you know the max and min value of EVERY SINGLE CONS!!!
        # Normalise all data for consensus between 0-1
        for column in df.columns:
            if 'VA__' in column:
                min_val = -1
                max_val = 1
            else:
                min_val = 0
                max_val = 1
            if column != 'p':
                df[column]= (df[column] - min_val) / (max_val - min_val)
        normalised[key] = copy.deepcopy(df)
    return normalised

def normalise_agents_time_series(sets):
    normalised = []
    for df in sets:
        ## This only works when you know the max and min value of EVERY SINGLE CONS!!!
        # Normalise all data for consensus between 0-1
        #df = df.astype(float)
        for column in df.columns:
            if 'VA__' in column:
                min_val = -1
                max_val = 1
            else:
                min_val = 0
                max_val = 1
            if column != 'p':
                df[column]= (df[column] - min_val) / (max_val - min_val)
        normalised.append(copy.deepcopy(df))
    return normalised

## CREDIT: https://nedbatchelder.com/blog/200712/human_sorting
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l

def parse_vector_cell(cell):
    """Parse a vector stored in a CSV cell.
    New synthetic consensus files store cons_pref/cons_act in a single
    cell, as a stringified Python-style list.
    """
    if isinstance(cell, list):
        return cell
    if pd.isna(cell):
        return []
    if isinstance(cell, str):
        # Remove the whitespace between the "[" and the first character, then the regex will match.
        # Otherwise, you end up with the string "[    mydecimalnumberhere" becoming [,mydecimalnumberhere], which is a syntax error
        cell = cell.strip('[]')
        return ast.literal_eval(','.join(re.sub(r'(?<=\S)(\s+)(?=-?\S)', ',', cell).splitlines()))
    return list(cell)


## CREDIT: https://nedbatchelder.com/blog/200712/human_sorting
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l

def parse_vector_cell(cell):
    """Parse a vector stored in a CSV cell.
    New synthetic consensus files store cons_pref/cons_act in a single
    cell, as a stringified Python-style list.
    """
    if isinstance(cell, list):
        return cell
    if pd.isna(cell):
        return []
    if isinstance(cell, str):
        # Remove the whitespace between the "[" and the first character, then the regex will match.
        # Otherwise, you end up with the string "[    mydecimalnumberhere" becoming [,mydecimalnumberhere], which is a syntax error
        cell = cell.strip('[]')
        return ast.literal_eval(','.join(re.sub(r'(?<=\S)(\s+)(?=-?\S)', ',', cell).splitlines()))
    return list(cell)

## CREDIT: https://nedbatchelder.com/blog/200712/human_sorting
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l

def parse_vector_cell(cell):
    """Parse a vector stored in a CSV cell.
    New synthetic consensus files store cons_pref/cons_act in a single
    cell, as a stringified Python-style list.
    """
    if isinstance(cell, list):
        return cell
    if pd.isna(cell):
        return []
    if isinstance(cell, str):
        # Remove the whitespace between the "[" and the first character, then the regex will match.
        # Otherwise, you end up with the string "[    mydecimalnumberhere" becoming [,mydecimalnumberhere], which is a syntax error
        cell = cell.strip('[]')
        return ast.literal_eval(','.join(re.sub(r'(?<=\S)(\s+)(?=-?\S)', ',', cell).splitlines()))
    return list(cell)


## CREDIT: https://nedbatchelder.com/blog/200712/human_sorting
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l

def parse_vector_cell(cell):
    """Parse a vector stored in a CSV cell.
    New synthetic consensus files store cons_pref/cons_act in a single
    cell, as a stringified Python-style list.
    """
    if isinstance(cell, list):
        return cell
    if pd.isna(cell):
        return []
    if isinstance(cell, str):
        # Remove the whitespace between the "[" and the first character, then the regex will match.
        # Otherwise, you end up with the string "[    mydecimalnumberhere" becoming [,mydecimalnumberhere], which is a syntax error
        cell = cell.strip('[]')
        return ast.literal_eval(','.join(re.sub(r'(?<=\S)(\s+)(?=-?\S)', ',', cell).splitlines()))
    return list(cell)
