from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

df = pd.read_csv("dfs.csv")
dfa =pd.read_csv("df.csv")
ln = LinearRegression()
tc = DecisionTreeClassifier()

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

""" Árvore de Decisão """

tr = DecisionTreeRegressor(criterion="squared_error", max_depth=20)
X = df[["idade", "SEXO", "ESPEC", "PROC_SOLIC", "MARCA_UTI"]]
X['SEXO'] = X["SEXO"].astype('category')
X['ESPEC'] = X["ESPEC"].astype('category')
X['PROC_SOLIC'] = X["PROC_SOLIC"].astype('category')
X['MARCA_UTI'] = X["MARCA_UTI"].astype('category')
y = df['COBRANCA'].astype('category')
X_train, X_test, y_train, y_test = train_test_split(X, y)

fit_tree = tr.fit(X_train, y_train)
fit_tree.get_n_leaves()

y_1 = fit_tree_sh.predict(x_test)
# from sklearn.tree import export_graphviz 

# # export the decision tree to a tree.dot file
# # for visualizing the plot easily anywhere
export_graphviz(tr, out_file ='tree2.dot',
               feature_names =X.keys())
import graphviz
graphviz.render(engine = 'dot', filepath='tree2.dot', outfile='tree2.pdf')

# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
# plt.plot(x_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
# # plt.plot(x_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()

""" VALORES """

X = dfa[["idade", "SEXO", "ESPEC", "PROC_SOLIC", "MARCA_UTI"]]
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




