# QuICT Command Line Interface

## How to add quict into bash command
```
# Step 1, using which python get the path of bin folder, end with bin.
which python

# Step 2, cp quict.py file into the path of bin folder.
cp QuICT/cloud/cli/quict.py $PATH_TO_BIN

# Step 3, rename quict.py with quict in bin folder.
mv $PATH_TO_BIN/quict.py $PATH_TO_BIN/quict
```

## How to use quict CLI
```
quict -h
```

### How to use quict circuit
```
quict circuit -h
```

### How to use quict local mode
```
quict local job -h
```

### How to use quict remote mode
```
# Login first
quict login [name] [password]

quict remote job -h
```
