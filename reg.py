from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import pickle

df = pd.read_csv("dfs.csv")
dfa =pd.read_csv("df.csv")
ln = LinearRegression()
# lo = LogisticRegression()

# X = df[["idade", "PROC_SOLIC"]]
# y = df["ESPEC"]
# lo.fit(X, y)
# lo.score(X, y)

# with open("ESPEC_<_idade_PROC_SOLIC", 'wb') as file:
#     pickle.dump(lo, file)

# paciente = pd.Series({
#     'idade': 19,
#     'PROC_SOLIC': 408060344
# })


# ndf = pd.DataFrame(paciente).transpose()
# lo.predict(ndf)


""" VALORES """

X = dfa[["idade", "SEXO"]]
y = dfa['VAL_SH']
                    
fit_val_sh = ln.fit(X, y)
fit_val_sh.score(X, y)

with open("VAL_SH_<_idade_SEXO.pickle", 'wb') as file2:
    pickle.dump(ln, file2)



# VAL SP
y = dfa['VAL_SP']
                    
fit_val_sh = ln.fit(X, y)
fit_val_sh.score(X, y)

with open("VAL_SP_<_idade_SEXO.pickle", 'wb') as file2:
    pickle.dump(ln, file2)
    



# VAL SP
y = dfa['VAL_UTI']
                    
fit_val_sh = ln.fit(X, y)
fit_val_sh.score(X, y)

with open("VAL_UTI_<_idade_SEXO.pickle", 'wb') as file2:
    pickle.dump(ln, file2)