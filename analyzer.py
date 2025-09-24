import pandas as pd
from datetime import datetime, time


def load_and_prepare_data(file_path):
    """
    IMPROVED: Membaca file CSV transaksi dan menyiapkan data untuk analisis DD.
    
    Perbaikan:
    - Validasi data yang lebih robust
    - Handling missing values dan data kotor
    - Error reporting yang lebih detail
    - Data type enforcement yang lebih baik
    
    Features:
    - Normalisasi kolom dan tipe data
    - Ubah trx_on menjadi datetime
    - Urutkan data sesuai tanggal transaksi
    - Tambah kolom entry_gate dan exit_gate untuk analisis
    """
    # Baca file CSV dengan error handling
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Gagal membaca file CSV: {str(e)}")
    
    if df.empty:
        raise ValueError("File CSV kosong atau tidak memiliki data")
    
    # Pastikan semua kolom yang dibutuhkan ada
    expected_cols = [
        "trx", "trx_on", "balance_before_int", "fare_int", "balance_int",
        "status_var", "deduct_boo", "desc_var", "shelter_code", "terminal_name_var",
        "card_type_id_int", "jenis_kartu", "card_number_var", "terminal_code",
        "issuer_type_var", "interop_id_int"
    ]

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom hilang di file: {missing_cols}")

    # Cleaning dan validasi data
    initial_rows = len(df)
    
    # Filter out rows dengan data critical yang null
    df = df.dropna(subset=["trx", "trx_on", "card_number_var"])
    
    # Ubah trx_on jadi datetime dengan error handling
    df["trx_on"] = pd.to_datetime(df["trx_on"], errors="coerce")
    
    # Remove rows dengan timestamp invalid
    df = df.dropna(subset=["trx_on"])
    
    # Fill missing numeric values dengan 0
    numeric_cols = ["balance_before_int", "fare_int", "balance_int", "card_type_id_int", "interop_id_int"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Normalisasi tipe data kolom boolean (deduct_boo → bool)
    df["deduct_boo"] = df["deduct_boo"].astype(str).str.upper().map({"TRUE": True, "FALSE": False, "NAN": False}).fillna(False)
    
    # Fill missing string values
    string_cols = ["status_var", "desc_var", "shelter_code", "terminal_name_var", "jenis_kartu", "terminal_code", "issuer_type_var"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("UNKNOWN")
    
    # Tambahkan kolom 'entry_gate' dan 'exit_gate' berdasarkan jenis transaksi
    df["entry_gate"] = df["terminal_name_var"].where(df["trx"].str.contains(r"\[IN\]", na=False))
    df["exit_gate"] = df["terminal_name_var"].where(df["trx"].str.contains(r"\[OUT\]", na=False))

    # Urutkan sesuai tanggal transaksi dan card
    df = df.sort_values(by=["card_number_var", "trx_on"]).reset_index(drop=True)
    
    # Report data cleaning results
    cleaned_rows = len(df)
    if initial_rows != cleaned_rows:
        print(f"Data cleaning: {initial_rows} → {cleaned_rows} rows ({initial_rows - cleaned_rows} rows dihapus)")

    if cleaned_rows == 0:
        raise ValueError("Tidak ada data valid setelah cleaning")

    return df


def is_payment_transaction(row):
    """
    IMPROVED: Enhanced validation untuk memastikan transaksi benar-benar memotong saldo.
    
    Perbaikan:
    - Validasi lebih ketat terhadap data null/invalid
    - Penanganan edge cases (saldo negatif, fare 0, etc)
    - Konsistensi mathematical validation
    
    Parameters:
        row (Series): Baris transaksi
    
    Returns:
        bool: True jika transaksi benar memotong saldo
    """
    try:
        # Cek null values
        if pd.isnull(row["deduct_boo"]) or pd.isnull(row["fare_int"]) or \
           pd.isnull(row["balance_before_int"]) or pd.isnull(row["balance_int"]):
            return False
        
        # Convert to proper types
        deduct_flag = bool(row["deduct_boo"])
        fare = int(row["fare_int"]) if row["fare_int"] != '' else 0
        balance_before = int(row["balance_before_int"]) if row["balance_before_int"] != '' else 0
        balance_after = int(row["balance_int"]) if row["balance_int"] != '' else 0
        
        # Enhanced validation
        is_deduct_true = deduct_flag == True
        has_fare = fare > 0
        balance_reduced = balance_before > balance_after
        
        # Mathematical consistency check
        actual_deduction = balance_before - balance_after
        
        # Toleransi untuk floating point precision atau rounding
        is_mathematically_consistent = abs(actual_deduction - fare) <= 1
        
        return (
            is_deduct_true and 
            has_fare and 
            balance_reduced and 
            is_mathematically_consistent
        )
        
    except (ValueError, TypeError, AttributeError):
        # Jika ada error dalam parsing data, anggap bukan payment
        return False


def is_subsidi_time(trx_time):
    """
    IMPROVED: Cek apakah transaksi terjadi di jam subsidi pagi (05:00-07:00).
    Pada jam ini tarif seharusnya 2000, bukan 3500.
    
    Perbaikan:
    - Validasi input yang lebih robust
    - Handling timezone dan edge cases
    
    Parameters:
        trx_time (datetime): Waktu transaksi
    
    Returns:
        bool: True jika di jam subsidi, False jika tidak
    """
    if pd.isnull(trx_time):
        return False
    
    try:
        # Pastikan ini datetime object
        if not isinstance(trx_time, (pd.Timestamp, datetime)):
            return False
        
        hour = trx_time.hour
        
        # Subsidi berlaku 05:00:00 - 06:59:59
        # (tidak termasuk 07:00:00 ke atas)
        return (hour == 5) or (hour == 6)
        
    except (AttributeError, ValueError):
        return False


def is_integration_fare(fare_int):
    """
    Cek apakah tarif termasuk tarif integrasi (> 3500).
    
    Parameters:
        fare_int (int): Jumlah tarif
    
    Returns:
        bool: True jika tarif integrasi
    """
    return fare_int > 3500


def normalize_terminal_name(terminal_name):
    """
    Normalisasi nama terminal untuk perbandingan.
    
    Parameters:
        terminal_name (str): Nama terminal asli
    
    Returns:
        str: Nama terminal yang dinormalisasi
    """
    if pd.isnull(terminal_name):
        return ""
    
    terminal_name = str(terminal_name).strip().upper()
    return terminal_name


def is_same_terminal(terminal1, terminal2):
    """
    Cek apakah dua terminal adalah terminal yang sama.
    
    Parameters:
        terminal1 (str): Terminal pertama
        terminal2 (str): Terminal kedua
    
    Returns:
        bool: True jika terminal sama
    """
    norm1 = normalize_terminal_name(terminal1)
    norm2 = normalize_terminal_name(terminal2)
    
    return norm1 == norm2


def get_trips_for_card(df, current_idx):
    """
    REVISI: Deteksi perjalanan yang akurat sesuai logika TransJakarta.
    
    Logika:
    - GATE IN → mulai perjalanan baru
    - TOB IN dalam trip yang sama (dalam 4 jam)
    - UNBLOKIR → pembayaran untuk perjalanan sebelumnya
    - OUT → akhir perjalanan
    
    Parameters:
        df (DataFrame): DataFrame semua transaksi
        current_idx (int): Index transaksi yang sedang dianalisis
    
    Returns:
        list: List berisi dict dengan info setiap trip
    """
    current_row = df.loc[current_idx]
    current_card = current_row["card_number_var"]
    
    # Ambil semua transaksi kartu sampai index saat ini
    card_transactions = df[
        (df["card_number_var"] == current_card) &
        (df.index <= current_idx)
    ].sort_values(by="trx_on").reset_index()
    
    trips = []
    current_trip = None
    MAX_TRIP_HOURS = 4  # Maksimal durasi perjalanan 4 jam
    
    for _, transaction in card_transactions.iterrows():
        original_idx = transaction['index']
        trx_type = transaction["trx"]
        trx_time = transaction["trx_on"]
        
        # 1. GATE IN = mulai perjalanan baru
        if "GATE" in trx_type and "[IN]" in trx_type:
            # Tutup trip sebelumnya jika ada
            if current_trip and not current_trip.get("is_completed"):
                current_trip["is_completed"] = True
                trips.append(current_trip)
            
            # Mulai trip baru
            current_trip = {
                "trip_id": len(trips) + 1,
                "start_idx": original_idx,
                "start_time": trx_time,
                "start_type": "GATE_IN",
                "transactions": [],
                "payments_count": 0,
                "is_paid": False,
                "is_completed": False
            }
            
            # Tambahkan transaksi ke trip
            current_trip["transactions"].append({
                "idx": original_idx,
                "type": trx_type,
                "time": trx_time,
                "is_payment": is_payment_transaction(transaction)
            })
            
            # Cek apakah GATE IN ini juga payment
            if is_payment_transaction(transaction):
                current_trip["payments_count"] += 1
                current_trip["is_paid"] = True
        
        # 2. TOB IN dalam trip yang sama
        elif current_trip and "TOB" in trx_type and "[IN]" in trx_type:
            time_gap_hours = (trx_time - current_trip["start_time"]).total_seconds() / 3600
            
            if time_gap_hours <= MAX_TRIP_HOURS:
                # Masih dalam trip yang sama
                current_trip["transactions"].append({
                    "idx": original_idx,
                    "type": trx_type,
                    "time": trx_time,
                    "is_payment": is_payment_transaction(transaction)
                })
                
                if is_payment_transaction(transaction):
                    current_trip["payments_count"] += 1
                    current_trip["is_paid"] = True
        
        # 3. UNBLOKIR - tidak masuk ke trip, diproses terpisah
        elif "UNBLOKIR" in trx_type:
            # UNBLOKIR tidak masuk ke current_trip
            # Akan dianalisis di is_double_deduct()
            pass
        
        # 4. OUT - akhiri trip
        elif current_trip and "[OUT]" in trx_type:
            current_trip["transactions"].append({
                "idx": original_idx,
                "type": trx_type,
                "time": trx_time,
                "is_payment": is_payment_transaction(transaction)
            })
            
            if is_payment_transaction(transaction):
                current_trip["payments_count"] += 1
                current_trip["is_paid"] = True
            
            current_trip["is_completed"] = True
            current_trip["end_time"] = trx_time
    
    # Simpan trip terakhir
    if current_trip:
        # Auto-complete jika sudah lewat batas waktu
        if not current_trip["is_completed"]:
            current_time = current_row["trx_on"]
            time_gap = (current_time - current_trip["start_time"]).total_seconds() / 3600
            if time_gap > MAX_TRIP_HOURS:
                current_trip["is_completed"] = True
        
        trips.append(current_trip)
    
    return trips


def find_last_unpaid_trip(df, current_idx):
    """
    Mencari perjalanan terakhir yang belum dibayar.
    UNBLOKIR bisa membayar trip ini.
    
    Parameters:
        df (DataFrame): DataFrame semua transaksi
        current_idx (int): Index transaksi yang sedang dianalisis
    
    Returns:
        dict or None: Info trip yang belum terbayar
    """
    trips = get_trips_for_card(df, current_idx)
    
    # Cari dari trip terbaru ke lama
    for trip in reversed(trips):
        # Skip trip yang mengandung transaksi current
        contains_current = any(t["idx"] == current_idx for t in trip["transactions"])
        if contains_current:
            continue
        
        # Trip belum dibayar jika payments_count = 0
        if trip["payments_count"] == 0:
            return trip
    
    return None


def is_double_deduct(df, idx):
    """
    REVISI: Logika deteksi DD sesuai dengan penjelasan TransJakarta yang benar.
    
    Logika DD:
    1. UNBLOKIR tanpa trip sebelumnya yang perlu dibayar = DD
    2. UNBLOKIR bersamaan waktu dengan IN berikutnya = DD  
    3. Payment ke-2+ dalam 1 trip yang sama = DD
    4. Tarif subsidi salah (3500 di jam 05:00-07:00) = DD
    5. UNBLOKIR membayar trip subsidi dengan tarif 3500 = DD
    
    Parameters:
        df (DataFrame): DataFrame semua transaksi
        idx (int): Index baris transaksi yang sedang dianalisis
    
    Returns:
        tuple: (is_dd, refund_amount, detected_step, is_integration)
    """
    try:
        row = df.loc[idx]
        
        # Hanya cek transaksi yang benar memotong saldo
        if not is_payment_transaction(row):
            return False, 0, "Bukan transaksi pembayaran", False
        
        is_integration = is_integration_fare(row["fare_int"])
        
        # === LOGIKA UNBLOKIR ===
        if "UNBLOKIR" in row["trx"]:
            # Cek apakah UNBLOKIR bersamaan waktu dengan IN berikutnya
            current_card = row["card_number_var"]
            current_time = row["trx_on"]
            
            # Cari IN dalam 5 menit setelah UNBLOKIR
            next_in = df[
                (df["card_number_var"] == current_card) &
                (df["trx_on"] > current_time) &
                (df["trx_on"] <= current_time + pd.Timedelta(minutes=5)) &
                (df["trx"].str.contains(r"\[IN\]", na=False))
            ]
            
            if not next_in.empty:
                gap_seconds = (next_in.iloc[0]["trx_on"] - current_time).total_seconds()
                return True, row["fare_int"], f"DD: UNBLOKIR bersamaan dengan IN (gap {gap_seconds:.0f}s)", is_integration
            
            # Cari apakah ada trip sebelumnya yang belum dibayar
            unpaid_trip = find_last_unpaid_trip(df, idx)
            
            if unpaid_trip:
                # Ada trip yang belum dibayar - UNBLOKIR sah, tapi cek subsidi
                if is_subsidi_time(unpaid_trip["start_time"]) and row["fare_int"] == 3500:
                    return True, 1500, f"DD Subsidi: UNBLOKIR bayar trip subsidi idx {unpaid_trip['start_idx']}, seharusnya 2000", is_integration
                else:
                    return False, 0, f"Sah: UNBLOKIR bayar trip idx {unpaid_trip['start_idx']}", is_integration
            else:
                # Tidak ada trip yang perlu dibayar - DD
                return True, row["fare_int"], "DD: UNBLOKIR tanpa trip yang perlu dibayar", is_integration
        
        # === LOGIKA IN/OUT dalam trip ===
        trips = get_trips_for_card(df, idx)
        current_trip = None
        payment_order_in_trip = 0
        
        # Cari trip yang mengandung transaksi ini
        for trip in trips:
            for i, transaction in enumerate(trip["transactions"]):
                if transaction["idx"] == idx:
                    current_trip = trip
                    # Hitung urutan payment dalam trip
                    payments_before = sum(1 for t in trip["transactions"][:i+1] if t["is_payment"])
                    payment_order_in_trip = payments_before
                    break
            if current_trip:
                break
        
        if current_trip and payment_order_in_trip > 1:
            # Payment ke-2, ke-3, dst dalam trip yang sama = DD
            return True, row["fare_int"], f"DD: Payment ke-{payment_order_in_trip} dalam trip idx {current_trip['start_idx']}", is_integration
        
        # === CEK SUBSIDI untuk payment normal ===
        if is_subsidi_time(row["trx_on"]) and row["fare_int"] == 3500:
            return True, 1500, "DD Subsidi: Jam 05:00-07:00 seharusnya tarif 2000", is_integration
        
        return False, 0, "Sah: Payment normal", is_integration
        
    except Exception as e:
        return False, 0, f"Error: {str(e)}", False


def analyze_all_transactions(df, debug_mode=False):
    """
    IMPROVED: Menganalisis semua transaksi dalam DataFrame untuk deteksi Double Deduct.
    
    Perbaikan:
    - Progress tracking untuk dataset besar
    - Debug mode untuk troubleshooting
    - Error handling per transaksi
    
    Parameters:
        df (DataFrame): DataFrame transaksi yang sudah dipersiapkan
        debug_mode (bool): Jika True, akan print progress dan debug info
    
    Returns:
        DataFrame: DataFrame dengan kolom analisis DD
    """
    df["is_dd"] = False
    df["dd_refund"] = 0
    df["dd_step"] = ""
    df["is_integration"] = False
    
    total_rows = len(df)
    processed = 0
    errors = 0

    if debug_mode:
        print(f"Memulai analisis {total_rows:,} transaksi...")

    for idx in df.index:
        try:
            is_dd, refund_amount, detected_step, is_integration = is_double_deduct(df, idx)

            df.at[idx, "is_dd"] = is_dd
            df.at[idx, "dd_refund"] = refund_amount
            df.at[idx, "dd_step"] = detected_step
            df.at[idx, "is_integration"] = is_integration
            
            processed += 1
            
            # Progress report setiap 1000 transaksi
            if debug_mode and processed % 1000 == 0:
                progress_pct = (processed / total_rows) * 100
                dd_found = df.loc[:idx, "is_dd"].sum()
                print(f"Progress: {processed:,}/{total_rows:,} ({progress_pct:.1f}%) - DD ditemukan: {dd_found}")

        except Exception as e:
            errors += 1
            if debug_mode:
                print(f"Error pada idx {idx}: {str(e)}")
            
            # Set default values untuk error cases
            df.at[idx, "is_dd"] = False
            df.at[idx, "dd_refund"] = 0
            df.at[idx, "dd_step"] = f"Error: {str(e)}"
            df.at[idx, "is_integration"] = False

    if debug_mode or errors > 0:
        dd_count = df["is_dd"].sum()
        total_refund = df["dd_refund"].sum()
        print(f"\n=== HASIL ANALISIS ===")
        print(f"Total transaksi: {total_rows:,}")
        print(f"Berhasil diproses: {processed:,}")
        print(f"Error: {errors}")
        print(f"DD ditemukan: {dd_count:,} ({(dd_count/total_rows)*100:.2f}%)")
        print(f"Total refund: Rp {total_refund:,.0f}")

    return df


def get_dd_summary(df):
    """
    Membuat ringkasan hasil analisis Double Deduct dengan pembagian refund TJ dan JLI.
    
    Parameters:
        df (DataFrame): DataFrame hasil analisis
    
    Returns:
        dict: Ringkasan statistik DD
    """
    dd_transactions = df[df["is_dd"] == True]
    integration_transactions = df[df["is_integration"] == True]
    
    # Pembagian refund berdasarkan card_type_id_int
    # TJ: 1-5, JLI: 6-25
    dd_tj = dd_transactions[dd_transactions["card_type_id_int"].between(1, 5)]
    dd_jli = dd_transactions[dd_transactions["card_type_id_int"].between(6, 25)]
    
    summary = {
        "total_transactions": len(df),
        "total_dd_cases": len(dd_transactions),
        "total_refund_amount": df["dd_refund"].sum(),
        "total_refund_tj": dd_tj["dd_refund"].sum(),
        "total_refund_jli": dd_jli["dd_refund"].sum(),
        "total_integration_fares": len(integration_transactions),
        "dd_percentage": (len(dd_transactions) / len(df)) * 100 if len(df) > 0 else 0,
        "integration_percentage": (len(integration_transactions) / len(df)) * 100 if len(df) > 0 else 0,
        "dd_by_card": dd_transactions.groupby("card_number_var")["dd_refund"].sum().to_dict(),
        "dd_by_type": {}
    }
        
    # Analisis per jenis transaksi
    for trx_type in ["GATE", "TOB", "UNBLOKIR"]:
        type_dd = dd_transactions[dd_transactions["trx"].str.contains(trx_type)]
        summary["dd_by_type"][trx_type] = {
            "cases": len(type_dd),
            "total_refund": type_dd["dd_refund"].sum()
        }
    
    return summary