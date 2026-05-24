
import time
import math
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# SETUP: Αρχικοποίηση Spark Session & Ρύθμιση Πόρων
spark = SparkSession.builder \
    .appName("TriadicJoin_MultiWay") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

CORES = spark.sparkContext.defaultParallelism
NUM_REDUCERS = CORES * 2
spark.conf.set("spark.sql.shuffle.partitions", str(NUM_REDUCERS))

# ΔΗΜΙΟΥΡΓΙΑ ΔΕΔΟΜΕΝΩΝ (Native JVM Generation)
# Η εκφώνηση ζητάει όλες οι αρχικές σχέσεις (A, B, C) να έχουν το ίδιο μέγεθος.
N = 100000  # 1 Εκατομμύριο εγγραφές ανά πίνακα

print(f"Δημιουργία πινάκων A, B, C με {N:,} ")

# Πίνακας A(x, y) - Το x είναι μοναδικό ID, το y είναι τυχαίο κλειδί σύνδεσης (domain 0-100k)
df_A = spark.range(N) \
    .select(F.col("id").alias("x"), (F.rand() * 100000).cast("int").alias("y"))

# Πίνακας B(y, z) - Το y συνδέεται με το A, το z συνδέεται με το C
df_B = spark.range(N) \
    .select((F.rand() * 100000).cast("int").alias("y"), (F.rand() * 100000).cast("int").alias("z"))

# Πίνακας C(z, w) - Το z συνδέεται με το B, το w είναι απλό πεδίο
df_C = spark.range(N) \
    .select((F.rand() * 100000).cast("int").alias("z"), (F.rand() * 100000).cast("int").alias("w"))

# Κάνουμε cache για να μετρήσουμε μόνο τον χρόνο των αλγορίθμων και όχι της δημιουργίας
df_A.cache(); df_B.cache(); df_C.cache()
df_A.count(); df_B.count(); df_C.count()

# ΑΛΓΟΡΙΘΜΟΣ 1: ΑΠΕΥΘΕΙΑΣ ΤΡΙΑΔΙΚΗ ΣΥΝΔΕΣΗ (Shares Algorithm / Afrati et al.)
# Αντί να ενώσουμε το A με το B (κάνοντας ένα ενδιάμεσο τεράστιο αποτέλεσμα) και 
# μετά να ενώσουμε το C, τα στέλνουμε όλα ΜΑΖΙ στους Reducers (1-step MapReduce).
# Έχουμε 2 μεταβλητές σύνδεσης: το `y` και το `z`. Πρέπει να στήσουμε ένα 2D Grid
# από κάδους (reducers), μεγέθους s_y * s_z = NUM_REDUCERS.
def direct_3way_join(A, B, C, reducers):
    # Βρίσκουμε τις διαστάσεις του Grid. Αν έχουμε 16 reducers, τότε 4x4.
    s_y = max(1, int(math.sqrt(reducers)))
    s_z = reducers // s_y
    actual_reducers = s_y * s_z  # Μπορεί να διαφέρει ελάχιστα από το αρχικό λόγω ακεραίων
    
    # ΠΩΣ ΔΙΑΝΕΜΟΥΜΕ ΤΑ ΔΕΔΟΜΕΝΑ (The Mapping Phase):
    # 1. Πίνακας A(x,y): Ξέρει το 'y', άρα ξέρει τη συντεταγμένη h_y = y % s_y.
    # Όμως ΔΕΝ ξέρει το 'z'. Άρα για να είναι σίγουρο ότι θα συναντήσει τα σωστά B και C,
    # πρέπει να στείλει αντίγραφο σε ΟΛΟΥΣ τους κάδους του άξονα z (από 0 έως s_z-1).
    A_mapped = A.withColumn("h_y", F.col("y") % s_y) \
                .withColumn("h_z", F.explode(F.expr(f"sequence(0, {s_z - 1})")))
                
    # 2. Πίνακας C(z,w): Ξέρει το 'z', άρα ξέρει τη συντεταγμένη h_z = z % s_z.
    # Δεν ξέρει το 'y', άρα κάνει το αντίστροφο: στέλνει αντίγραφο σε ΟΛΟΥΣ τους άξονες y.
    C_mapped = C.withColumn("h_z", F.col("z") % s_z) \
                .withColumn("h_y", F.explode(F.expr(f"sequence(0, {s_y - 1})")))
                
    # 3. Πίνακας B(y,z): Αυτός Ξέρει ΚΑΙ το 'y' ΚΑΙ το 'z'.
    # Δεν χρειάζεται καμία αντιγραφή (explode). Ξέρει ακριβώς το κελί του στο Grid.
    B_mapped = B.withColumn("h_y", F.col("y") % s_y) \
                .withColumn("h_z", F.col("z") % s_z)

    # SHUFFLE PHASE: Στέλνουμε τα πάντα μέσω δικτύου βάσει των συντεταγμένων (h_y, h_z)
    A_rep = A_mapped.repartition(actual_reducers, "h_y", "h_z")
    B_rep = B_mapped.repartition(actual_reducers, "h_y", "h_z")
    C_rep = C_mapped.repartition(actual_reducers, "h_y", "h_z")
    
    # LOCAL JOIN: Τώρα που όλες οι εγγραφές είναι στους σωστούς reducers,
    # τις αφήνουμε να κουμπώσουν. Το Spark θα κάνει τοπικά joins χωρίς καθόλου επιπλέον δίκτυο
    # Ενώνουμε τα A και B χρησιμοποιώντας ΚΑΙ τις συντεταγμένες για να μην υπάρξει cross-talk.
    join_AB = A_rep.join(B_rep, ["h_y", "h_z", "y"])
    final_join = join_AB.join(C_rep, ["h_y", "h_z", "z"])
    
    return final_join.select("x", "y", "z", "w")


# ΑΛΓΟΡΙΘΜΟΣ 2: 2 ΔΙΑΔΟΧΙΚΕΣ ΔΥΑΔΙΚΕΣ ΣΥΝΔΕΣΕΙΣ 
# Εδώ ακολουθούμε τον παραδοσιακό, δισταδιακό τρόπο.
# Το Spark θα εκτελέσει το A ⋈ B, θα πάρει το προσωρινό αποτέλεσμα, θα το κάνει 
# ξανά Shuffle στο δίκτυο, και θα το ενώσει με το C. Είναι O(n) κώδικας αλλά έχει
# 2 κύκλους I/O (γράψε-διάβασε από τον δίσκο) κατά τη διάρκεια του Shuffle.
def sequential_binary_joins(A, B, C):
    # Ενώνουμε το A με το B στο κοινό τους κλειδί 'y'
    temp_AB = A.join(B, "y")
    # Το αποτέλεσμα έχει στήλες (y, x, z). Τώρα το ενώνουμε με το C στο 'z'.
    final_join = temp_AB.join(C, "z")
    return final_join.select("x", "y", "z", "w")


# ΑΛΓΟΡΙΘΜΟΣ 3: SQL ΕΡΩΤΗΜΑ (Catalyst Optimizer)
# Αντί να πούμε στο Spark "πώς" να το κάνει, του λέμε "τι" θέλουμε μέσω SQL.
# Ο Catalyst Optimizer θα διαβάσει τα στατιστικά των πινάκων, θα αποφασίσει μόνος
# του τη σειρά των Joins (π.χ. μήπως συμφέρει να ενώσει πρώτα το B με το C;)
# και θα επιλέξει αυτόματα SortMergeJoin ή BroadcastJoin.
def sql_3way_join(spark, A, B, C):
    A.createOrReplaceTempView("A")
    B.createOrReplaceTempView("B")
    C.createOrReplaceTempView("C")
    
    query = """
        SELECT A.x, B.y, B.z, C.w
        FROM A
        JOIN B ON A.y = B.y
        JOIN C ON B.z = C.z
    """
    return spark.sql(query)


# ΕΚΤΕΛΕΣΗ ΚΑΙ ΑΞΙΟΛΟΓΗΣΗ ΧΡΟΝΩΝ
print("\n--- 1. Απευθείας Τριαδική Σύνδεση (Shares Algorithm) ---")
t0 = time.time()
res1 = direct_3way_join(df_A, df_B, df_C, NUM_REDUCERS)
count1 = res1.count()
time1 = time.time() - t0
print(f"Αποτέλεσμα: {count1:,} γραμμές | Χρόνος: {time1:.2f} δευτερόλεπτα")

print("\n--- 2. 2 Διαδοχικές Δυαδικές Συνδέσεις (Sequential API) ---")
t0 = time.time()
res2 = sequential_binary_joins(df_A, df_B, df_C)
count2 = res2.count()
time2 = time.time() - t0
print(f"Αποτέλεσμα: {count2:,} γραμμές | Χρόνος: {time2:.2f} δευτερόλεπτα")

print("\n--- 3. Ερώτημα SQL (Catalyst Optimizer) ---")
t0 = time.time()
res3 = sql_3way_join(spark, df_A, df_B, df_C)
count3 = res3.count()
time3 = time.time() - t0
print(f"Αποτέλεσμα: {count3:,} γραμμές | Χρόνος: {time3:.2f} δευτερόλεπτα")

spark.stop()