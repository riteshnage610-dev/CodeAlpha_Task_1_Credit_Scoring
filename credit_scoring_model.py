import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay

SEED=42
os.makedirs("outputs", exist_ok=True)

def make_data(n=2000):
    rng=np.random.default_rng(SEED)
    age=rng.integers(21,66,n)
    income=np.clip(rng.normal(55000,18000,n),18000,150000)
    debt=np.clip(rng.normal(18000,10000,n),1000,80000)
    history=np.clip(age-rng.integers(18,30,n)+rng.normal(0,2,n),0,35)
    late=np.clip(rng.poisson(1.2,n),0,10)
    loans=np.clip(rng.poisson(1.5,n),0,6)
    savings=np.clip(rng.normal(30000,18000,n),500,120000)
    employment=np.clip(rng.normal(7,5,n),0,35)
    expenses=np.clip(rng.normal(2600,900,n),700,7000)
    score=.000018*income-2.4*(debt/income)+.055*history-.32*late-.18*loans+.000012*savings+.035*employment-.00008*expenses+rng.normal(0,.7,n)
    y=(1/(1+np.exp(-score))>=.52).astype(int)
    return pd.DataFrame({"age":age,"annual_income":income,"total_debt":debt,"credit_history_years":history,"late_payments":late,"existing_loans":loans,"savings":savings,"employment_years":employment,"monthly_expenses":expenses,"creditworthy":y})

df=make_data()
df.to_csv("outputs/credit_scoring_dataset.csv",index=False)
X=df.drop(columns="creditworthy"); y=df.creditworthy
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
models={
"Logistic Regression":Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(max_iter=2000))]),
"Decision Tree":Pipeline([("imp",SimpleImputer(strategy="median")),("model",DecisionTreeClassifier(max_depth=6,random_state=SEED))]),
"Random Forest":Pipeline([("imp",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=250,max_depth=8,random_state=SEED,n_jobs=-1))])
}
rows={}
for name,m in models.items():
    m.fit(Xtr,ytr); p=m.predict(Xte); prob=m.predict_proba(Xte)[:,1]
    rows[name]={"Accuracy":accuracy_score(yte,p),"Precision":precision_score(yte,p),"Recall":recall_score(yte,p),"F1-Score":f1_score(yte,p),"ROC-AUC":roc_auc_score(yte,prob)}
pd.DataFrame(rows).T.to_csv("outputs/model_comparison.csv")
best=models["Random Forest"]; pred=best.predict(Xte)
fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(confusion_matrix(yte,pred)); ax.set_title("Random Forest Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
cm=confusion_matrix(yte,pred)
for i in range(2):
    for j in range(2): ax.text(j,i,cm[i,j],ha="center",va="center")
fig.tight_layout(); fig.savefig("outputs/confusion_matrix.png",dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(7,5))
for name,m in models.items(): RocCurveDisplay.from_estimator(m,Xte,yte,name=name,ax=ax)
ax.set_title("Credit Scoring ROC Curves"); fig.tight_layout(); fig.savefig("outputs/roc_curves.png",dpi=180); plt.close(fig)
imp=pd.Series(best.named_steps["model"].feature_importances_,index=X.columns).sort_values()
imp.to_csv("outputs/feature_importance.csv")
fig,ax=plt.subplots(figsize=(7,5)); imp.plot.barh(ax=ax); ax.set_title("Random Forest Feature Importance"); fig.tight_layout(); fig.savefig("outputs/feature_importance.png",dpi=180); plt.close(fig)
print(pd.DataFrame(rows).T.round(4))
