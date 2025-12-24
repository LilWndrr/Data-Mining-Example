import pandas
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import os

# --- 1. Load Data ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'my_location_features.csv')
df = pandas.read_csv(csv_path)

# --- 2. Clean & Map Data ---
# Map Days
d_day = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
df['Day'] = df['Day'].map(d_day)

# Map Time Period (YENİ EKLENEN KISIM)
d_time_period = {'Morning':1, 'Afternoon':2, 'Evening':3, 'Night':4}
# Hata almamak için: Eğer reformating.py çalıştırılmadıysa ve sütun yoksa uyarı ver
if 'Time_Period' in df.columns:
    df['Time_Period'] = df['Time_Period'].map(d_time_period)
else:
    print("UYARI: 'Time_Period' sütunu bulunamadı! Lütfen önce reformating.py dosyasını çalıştırın.")
    exit()

# Map Activities
d_act = {'CAR':1, 'BUS':2, 'WALKING':3, 'CYCLING':4, 'RAIL':5}
# Ters sözlük (Grafiklerde isimleri yazmak için)
d_act_rev = {1:'CAR', 2:'BUS', 3:'WALKING', 4:'CYCLING', 5:'RAIL'}
df['Activity'] = df['Activity'].map(d_act)

# Drop rows with missing values
df = df.dropna()

# Select Features
features = ['Day', 'Distance_m', 'Duration_min', 'Speed_m_min', 'Time_Period']
X = df[features]
y = df['Activity']

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Train the Model ---
# Class_weight='balanced' ekledik (Dengesiz veri setini düzeltmek için önerim)
dtree = DecisionTreeClassifier(max_depth=5) 
dtree.fit(X_train, y_train)

# --- 5. Test the Model ---
predictions = dtree.predict(X_test)
score = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {score * 100:.2f}%")

# --- 6. GÖRSELLEŞTİRME (YENİ) ---

# A) Karmaşıklık Matrisi (Confusion Matrix)
# Hangi sınıfın hangi sınıfla karıştırıldığını gösterir.
fig, ax = plt.subplots(figsize=(10, 5))
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[d_act_rev[i] for i in sorted(d_act_rev.keys()) if i in y.unique()])
disp.plot(cmap='Blues', ax=ax)
plt.title(f'Confusion Matrix (Accuracy: {score*100:.1f}%)')
plt.show()

# B) Karar Ağacı Yapısı (Decision Tree Plot)
# Ağacın nasıl karar verdiğini gösterir (Rapor için harika bir görseldir)
plt.figure(figsize=(15, 10))
plot_tree(dtree, feature_names=features, class_names=[d_act_rev[i] for i in sorted(d_act_rev.keys()) if i in y.unique()], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

print("\n--- Classification Report Example ---")

print(classification_report(y_test, predictions, target_names=['CAR', 'BUS', 'WALKING', 'CYCLING', 'RAIL']))