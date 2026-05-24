
import time
import random
from math import sqrt
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import pyspark.sql.functions as F

#Ξεκινάμε το SparkSession με 4GB μνήμη για driver και executor
#Ονομάζουμε την εφαρμογή για ευκολότερη αναγνώριση στο Spark UI
#Δίνουμε 4 Gigabytes RAM στον Driver και στους Executors για να αποφύγουμε OOM errors με το μεγάλο dataset 
#Ομως, αν θέλετε να δοκιμάσετε με ακόμα μεγαλύτερα δεδομένα, μπορείτε να αυξήσετε αυτά τα όρια ανάλογα με τη διαθέσιμη μνήμη του συστήματός σας.
#Όταν θα τρέξει τοπικα και τα 8G ram καταναλώνονται απο το μηχάνημα αν ήμασταν σε πραγματικό cluster θα έπαιρνε 4 ο driver και 4 ο executor.
#To get or Create είναι το if not exists, create it, else get the existing one. Αν τρέξουμε το script πολλές φορές, δεν θα δημιουργεί νέα sessions αλλά θα χρησιμοποιεί το ίδιο, αποφεύγοντας έτσι περιττές καταναλώσεις πόρων.
spark = SparkSession.builder \
    .appName("AllPairsMatching_UltraFast") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
#Aποτρέπουμε το Spark από το να εμφανίζει περιττές πληροφορίες στο log, κρατώντας μόνο τα σημαντικά μηνύματα σφάλματος. Αυτό βοηθάει στο να έχουμε καθαρή έξοδο και να εστιάζουμε στα αποτελέσματα και τις μετρήσεις μας.
spark.sparkContext.setLogLevel("ERROR")
# Ο αριθμός των reducers καθορίζεται από το Spark αυτόματα βάσει του αριθμού των πυρήνων (cores) που έχει διαθέσιμα το σύστημα. Με την παράμετρο defaultParallelism, μπορούμε να πάρουμε αυτή την τιμή και να την χρησιμοποιήσουμε για να ρυθμίσουμε τον αριθμό των partitions στο shuffle, ώστε να βελτιστοποιήσουμε την απόδοση.
NUM_REDUCERS = spark.sparkContext.defaultParallelism

# Αποτρέπουμε το Spark από το να δημιουργεί 200 άσκοπα partitions στο Shuffle γιατί 200; Αυτό είναι το default του Spark για το shuffle, αλλά σε ένα μικρό dataset όπως το δικό μας, αυτό μπορεί να οδηγήσει σε υπερβολικό overhead. Ρυθμίζοντας τον αριθμό των shuffle partitions ίσο με τον αριθμό των reducers (cores) που έχουμε, μπορούμε να βελτιώσουμε σημαντικά την απόδοση του join.
#Σε ένα πραγματικό Production Cluster, όμως, ο χρυσός κανόνας δεν είναι 1 προς 1. Ο χρυσός κανόνας είναι 2x ή 3x των cores γιατί; Γιατί θέλουμε να έχουμε αρκετά partitions για να εκμεταλλευτούμε την παράλληλη επεξεργασία, αλλά όχι τόσα πολλά που να δημιουργούν υπερβολικό overhead. Σε ένα μικρό dataset, το 1 προς 1 μπορεί να είναι ιδανικό, αλλά σε μεγαλύτερα datasets, μπορεί να θέλουμε λίγο περισσότερα partitions για να διασφαλίσουμε ότι όλοι οι πυρήνες είναι απασχολημένοι.
spark.conf.set("spark.sql.shuffle.partitions", str(NUM_REDUCERS))
#Εδώ commented out το AQE γιατί σε ένα μικρό dataset όπως το δικό μας, μπορεί να μην κάνει μεγάλη διαφορά και μπορεί να προσθέσει λίγο overhead. Σε ένα πραγματικό cluster με μεγαλύτερα δεδομένα, όμως, το AQE μπορεί να είναι πολύ χρήσιμο για να βελτιστοποιήσει τα σχέδια εκτέλεσης δυναμικά, ειδικά αν δεν είμαστε σίγουροι για τον καλύτερο αριθμό partitions εκ των προτέρων.
'''
# Ο αριθμός των διαθέσιμων πυρήνων (cores) στο σύστημα (Driver/Cluster)
CORES = spark.sparkContext.defaultParallelism

# Σε production, θέλουμε 2-3 tasks ανά πυρήνα για καλύτερο Load Balancing (αποφυγή stragglers).
# Εδώ πολλαπλασιάζουμε επί 2.
NUM_REDUCERS = CORES * 2

# Ρυθμίζουμε τα partitions του shuffle για να αποφύγουμε το default 200.
spark.conf.set("spark.sql.shuffle.partitions", str(NUM_REDUCERS))

# Προαιρετικό "Cheat Code" (Για Spark 3.x): 
# Ενεργοποιούμε το Adaptive Query Execution (AQE).
# Αν εμείς κάνουμε λάθος στα partitions, το Spark τα ενώνει ή τα σπάει δυναμικά την ώρα που τρέχει!
spark.conf.set("spark.sql.adaptive.enabled", "true")
'''
#Εδώ λιγα λόγια για το dataset που θα χρησιμοποιήσουμε. Δημιουργούμε ένα τεράστιο dataset με 500.000 εγγραφές, όπου κάθε εγγραφή αντιπροσωπεύει μια ασθένεια και τον αριθμό των ασθενών που την έχουν. Οι ασθένειες επιλέγονται τυχαία από μια λίστα, και ο αριθμός των ασθενών είναι επίσης τυχαίος μεταξύ 10 και 5000. Αυτό μας δίνει ένα αρκετά μεγάλο dataset για να δοκιμάσουμε τις διαφορετικές προσεγγίσεις μας στο all-pairs matching.
disease_names = [
    "Covid-19", "Pneumonia",
    "Lung carcinoma", "Multiple sclerosis", "cystic fibrosis"
]

N = 10000
random.seed(42)
#Εδω δημιουργούμε τα δεδομένα αλλά είναι λάθος γιατί; γιατί τα list comprehensions πανε μονο στον driver και δημιουργούν ένα τεράστιο αντικείμενο στη μνήμη του driver, κάτι που μπορεί να οδηγήσει σε OOM errors. Σε ένα πραγματικό cluster, θα θέλαμε να δημιουργήσουμε τα δεδομένα με έναν πιο κατανεμημένο τρόπο, ίσως χρησιμοποιώντας RDDs ή DataFrames για να παράγουμε τα δεδομένα απευθείας στο cluster, αντί να τα δημιουργούμε όλα στον driver και μετά να τα στέλνουμε στο cluster.
raw_data = [
    (random.choice(disease_names) + "_" + str(i), random.randint(10, 5000))
    for i in range(N)
]
#Φτιαχνουμε το DataFrame με ένα αυστηρό schema.
# Λέγοντάς του ότι δεν υπάρχουν nulls, ο Catalyst απλά διαγράφει όλους τους ελέγχους null checks από το Physical Plan, επιταχύνοντας έτσι την εκτέλεση. Σε ένα πραγματικό σενάριο, αν γνωρίζουμε ότι τα δεδομένα μας είναι καθαρά και δεν περιέχουν nulls, αυτή η πληροφορία μπορεί να βοηθήσει το Spark να βελτιστοποιήσει περαιτέρω τα σχέδια εκτέλεσης.
# Αλλά σε σύνδετα δεδομένα, αν δεν είμαστε σίγουροι για την ποιότητα των δεδομένων, είναι καλύτερο να αφήσουμε το Spark να κάνει τους ελέγχους του για να αποφύγουμε απροσδόκητα σφάλματα κατά την εκτέλεση.
schema = StructType([
    StructField("disease", StringType(), False), 
    StructField("patients", IntegerType(), False)
])

df = spark.createDataFrame(raw_data, schema)
df.cache() # Κρατάμε το DataFrame στη μνήμη για να αποφύγουμε επαναλαμβανόμενα I/O κατά τις μετρήσεις μας. Αυτό είναι σημαντικό γιατί θα εκτελέσουμε πολλαπλές προσεγγίσεις στο ίδιο dataset, και θέλουμε να μετρήσουμε μόνο τον χρόνο του join, όχι τον χρόνο φόρτωσης των δεδομένων.
df.count() # Force action to load data in memory instantly αλλιώς θα το έκανε στο πρώτο join, και τότε θα μετρούσαμε και τον χρόνο φόρτωσης μαζί με τον χρόνο του join, κάτι που δεν θέλουμε.

'''
N = 500000  # Τώρα μπορείς να το κάνεις και 50.000.000 άφοβα!

disease_names = [
    "Covid-19", "Pneumonia",
    "Lung carcinoma", "Multiple sclerosis", "cystic fibrosis"
]

# Βήμα 1: Μετατρέπουμε τη μικρή Python list σε Spark Array Column
disease_array_col = F.array(*[F.lit(name) for name in disease_names])
num_diseases = len(disease_names)

# Βήμα 2: Γεννάμε τα δεδομένα Native (Κατανεμημένα από την αρχή)
# Η spark.range(N) δημιουργεί αστραπιαία μια στήλη "id" από το 0 έως το N-1
df = spark.range(N) \
    .withColumn(
        # Παίρνουμε ένα τυχαίο όνομα ασθένειας (1 έως 5)
        "disease_base", 
        F.element_at(disease_array_col, F.ceil(F.rand() * num_diseases).cast("int"))
    ) \
    .withColumn(
        # Ενώνουμε το όνομα με το id (π.χ. "Covid-19_1234")
        "disease", 
        F.concat(F.col("disease_base"), F.lit("_"), F.col("id").cast("string"))
    ) \
    .withColumn(
        # Παράγουμε τυχαίο αριθμό ασθενών από 10 έως 5000
        "patients", 
        F.round(F.rand() * (5000 - 10) + 10).cast("int")
    ) \
    .select("disease", "patients") # Κρατάμε μόνο τις στήλες που χρειαζόμαστε

# Βήμα 3: Caching & Action
df.cache()
print(f"Data generated successfully: {df.count()} rows")
'''


def naive_all_pairs_df(df):
    # Απευθείας Theta-Join αντί για crossJoin + filter
    return df.alias("a").join(df.alias("b"), F.col("a.disease") < F.col("b.disease")) \
             .selectExpr(
                 "a.disease as disease_A", 
                 "b.disease as disease_B", 
                 "abs(a.patients - b.patients) as patients_diff"
             )

# GROUP-BASED APPROACH (Afrati et al.)
def compute_num_groups(n, num_reducers):
    return max(2, int(sqrt(n / num_reducers)))

def group_based_all_pairs_df(df, num_reducers):
    n = df.count()
    g = compute_num_groups(n, num_reducers)
    
    # Αρχική ανάθεση της εγγενούς ομάδας (gid)
    df_g = df.withColumn("gid", F.abs(F.hash(F.col("disease"))) % g)
    
   
    # MAP PHASE Αντιγραφή δεδομένων για τα σωστά buckets
    # Πλευρά A: Η ομάδα 'i' πρέπει να συναντήσει όλες τις ομάδες 'j' >= 'i'
    # Δημιουργούμε ένα Array από το gid μέχρι το g-1, και κάνουμε explode
    left = df_g.withColumn("p_i", F.col("gid")) \
               .withColumn("p_j", F.explode(F.expr(f"sequence(gid, {g-1})"))) \
               .alias("a")
               
    # Πλευρά B: Η ομάδα 'j' πρέπει να συναντήσει όλες τις ομάδες 'i' <= 'j'
    # Δημιουργούμε ένα Array από το 0 μέχρι το gid, και κάνουμε explode
    right = df_g.withColumn("p_j", F.col("gid")) \
                .withColumn("p_i", F.explode(F.expr("sequence(0, gid)"))) \
                .alias("b")

    # ---------------------------------------------------------------------
    # SHUFFLE & REDUCE PHASE (Τοπικό Join)
    # ---------------------------------------------------------------------
    # Αναγκάζουμε το Spark να κάνει shuffle βάσει του κάδου (p_i, p_j)
    left_rep = left.repartition(num_reducers, "p_i", "p_j")
    right_rep = right.repartition(num_reducers, "p_i", "p_j")
    
    # Επειδή το Join τώρα είναι EQUI-JOIN (ισότητα στα keys), το Spark
    # τρέχει τον αστραπιαίο αλγόριθμο SortMergeJoin χωρίς nested loops!
    joined = left_rep.join(right_rep, ["p_i", "p_j"])

    # Φιλτράρισμα προστασίας Cross-Groups (όπως πριν)
    valid_pairs = joined.filter(
        (F.col("a.gid") < F.col("b.gid")) | 
        ((F.col("a.gid") == F.col("b.gid")) & (F.col("a.disease") < F.col("b.disease")))
    )
    
    return valid_pairs.selectExpr(
        "a.disease as disease_A", 
        "b.disease as disease_B", 
        "abs(a.patients - b.patients) as patients_diff"
    )

# ============================================================
# 5. SQL APPROACH
# ============================================================

def sql_all_pairs(spark, df):
    df.createOrReplaceTempView("diseases_table")
    return spark.sql("""
        SELECT
            a.disease                    AS disease_A,
            b.disease                    AS disease_B,
            ABS(a.patients - b.patients) AS patients_diff
        FROM diseases_table a
        JOIN diseases_table b
          ON a.disease < b.disease
    """)


# ============================================================
# 6. ΕΚΤΕΛΕΣΗ & ΜΕΤΡΗΣΗ ΧΡΟΝΩΝ
# ============================================================

print("Εκτέλεση SQL approach...")
t0 = time.time()
sql_result = sql_all_pairs(spark, df)
sql_count  = sql_result.count()
sql_time   = time.time() - t0

print("Εκτέλεση Naive approach...")
t0 = time.time()
naive_result = naive_all_pairs_df(df)
naive_count  = naive_result.count()
naive_time   = time.time() - t0

print("Εκτέλεση Group-based approach...")
t0 = time.time()
group_result = group_based_all_pairs_df(df, NUM_REDUCERS)
group_count  = group_result.count()
group_time   = time.time() - t0

print("\nSQL Query Plan (Spark Catalyst):")
sql_result.explain(extended=False)


# ============================================================
# 7. ΑΠΟΤΙΜΗΣΗ — Paper bounds
# ============================================================

n = N
g = compute_num_groups(n, NUM_REDUCERS)
reducers_naive = n * (n - 1) // 2
reducers_group = g * (g - 1) // 2 + g

print("\n" + "="*55)
print(f"  n={n} | r={NUM_REDUCERS} | g=√({n}/{NUM_REDUCERS})={g}")
print(f"  {'':20} {'Ζεύγη':>8} {'Χρόνος':>8} {'Reducers':>10}")
print("-"*55)
print(f"  {'Naive':<20} {naive_count:>8} {naive_time:>7.2f}s {reducers_naive:>10}")
print(f"  {'Group-based':<20} {group_count:>8} {group_time:>7.2f}s {reducers_group:>10}")
print(f"  {'SQL':<20} {sql_count:>8} {sql_time:>7.2f}s {'auto':>10}")
print("="*55)

spark.stop()