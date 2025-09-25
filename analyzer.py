import pandas as pd
from datetime import datetime, time
import re


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
    IMPROVED: Cek apakah transaksi terjadi di jam subsidi pagi (05:00-07:00 inclusive).
    Pada jam ini tarif seharusnya 2000, bukan 3500.
    
    Perbaikan:
    - Validasi input yang lebih robust
    - Handling timezone dan edge cases
    - Fix: Include jam 07:00:00 dalam periode subsidi
    
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
        minute = trx_time.minute
        second = trx_time.second
        
        # Subsidi berlaku 05:00:00 - 07:00:00 (inclusive)
        # Include jam 07:00:00 exactly, but not after
        if hour == 5 or hour == 6:
            return True
        elif hour == 7 and minute == 0 and second == 0:
            return True
        else:
            return False
        
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


def parse_terminal_name(terminal_name):
    """
    BARU: Parse nama terminal untuk ekstrak nama stasiun/terminal utama.
    
    Contoh:
    - "GATE 1 Salemba" → "Salemba"
    - "GATE 2 Salemba" → "Salemba"  
    - "TOB Halte Bundaran Senayan" → "Bundaran Senayan"
    
    Parameters:
        terminal_name (str): Nama terminal/gate lengkap
    
    Returns:
        str: Nama terminal yang sudah dinormalisasi
    """
    if pd.isnull(terminal_name) or terminal_name == "UNKNOWN":
        return ""
    
    terminal_name = str(terminal_name).strip().upper()
    
    # Pattern untuk ekstrak nama terminal utama
    # Hilangkan "GATE [angka]", "TOB", "HALTE", dll
    patterns = [
        r'^GATE\s+\d+\s+(.+)$',      # "GATE 1 Salemba" → "Salemba"
        r'^TOB\s+HALTE\s+(.+)$',     # "TOB HALTE Bundaran" → "Bundaran" 
        r'^TOB\s+(.+)$',             # "TOB Salemba" → "Salemba"
        r'^HALTE\s+(.+)$',           # "HALTE Salemba" → "Salemba"
    ]
    
    for pattern in patterns:
        match = re.match(pattern, terminal_name)
        if match:
            return match.group(1).strip()
    
    # Jika tidak cocok pattern, return terminal name asli
    return terminal_name


def is_same_terminal_group(terminal1, terminal2):
    """
    BARU: Cek apakah dua gate/terminal berada di terminal/stasiun yang sama.
    
    Parameters:
        terminal1 (str): Terminal pertama
        terminal2 (str): Terminal kedua
    
    Returns:
        bool: True jika di terminal yang sama
    """
    parsed1 = parse_terminal_name(terminal1)
    parsed2 = parse_terminal_name(terminal2)
    
    if not parsed1 or not parsed2:
        return False
    
    return parsed1 == parsed2


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
    Cek apakah dua terminal adalah terminal yang sama persis.
    
    Parameters:
        terminal1 (str): Terminal pertama
        terminal2 (str): Terminal kedua
    
    Returns:
        bool: True jika terminal sama persis
    """
    norm1 = normalize_terminal_name(terminal1)
    norm2 = normalize_terminal_name(terminal2)
    
    return norm1 == norm2


def get_trips_for_card(df, current_idx):
    """
    REVISI MAJOR: Deteksi perjalanan yang akurat sesuai logika TransJakarta yang benar.
    
    Logika Baru:
    1. GATE IN atau TOB IN → mulai perjalanan baru
    2. GATE OUT atau TOB OUT → akhiri perjalanan 
    3. Cross-platform OK: GATE IN → TOB OUT adalah 1 perjalanan
    4. Cross-platform OK: TOB IN → GATE OUT adalah 1 perjalanan  
    5. UNBLOKIR → tidak masuk trip, dianalisis terpisah
    6. OUT selalu mengakhiri trip, transaksi berikutnya adalah trip baru
    
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
        
        # 1. IN (GATE atau TOB) = mulai perjalanan baru
        if "[IN]" in trx_type and ("GATE" in trx_type or "TOB" in trx_type):
            # Tutup trip sebelumnya jika masih open (edge case)
            if current_trip and not current_trip.get("is_completed"):
                current_trip["is_completed"] = True
                trips.append(current_trip)
            
            # Mulai trip baru
            trip_type = "GATE_IN" if "GATE" in trx_type else "TOB_IN"
            current_trip = {
                "trip_id": len(trips) + 1,
                "start_idx": original_idx,
                "start_time": trx_time,
                "start_type": trip_type,
                "start_terminal": transaction["terminal_name_var"],
                "transactions": [],
                "payments_count": 0,
                "is_paid": False,
                "is_completed": False,
                "end_time": None,
                "end_type": None,
                "end_terminal": None
            }
            
            # Tambahkan transaksi ke trip
            current_trip["transactions"].append({
                "idx": original_idx,
                "type": trx_type,
                "time": trx_time,
                "terminal": transaction["terminal_name_var"],
                "is_payment": is_payment_transaction(transaction)
            })
            
            # Cek apakah IN ini juga payment
            if is_payment_transaction(transaction):
                current_trip["payments_count"] += 1
                current_trip["is_paid"] = True
        
        # 2. UNBLOKIR - tidak masuk ke trip, diproses terpisah
        elif "UNBLOKIR" in trx_type:
            # UNBLOKIR tidak masuk ke current_trip
            # Akan dianalisis di is_double_deduct()
            pass
        
        # 3. OUT (GATE atau TOB) = akhiri trip
        elif current_trip and "[OUT]" in trx_type and ("GATE" in trx_type or "TOB" in trx_type):
            # Tambahkan transaksi OUT ke trip
            current_trip["transactions"].append({
                "idx": original_idx,
                "type": trx_type,
                "time": trx_time,
                "terminal": transaction["terminal_name_var"],
                "is_payment": is_payment_transaction(transaction)
            })
            
            if is_payment_transaction(transaction):
                current_trip["payments_count"] += 1
                current_trip["is_paid"] = True
            
            # AKHIRI TRIP - ini yang penting!
            current_trip["is_completed"] = True
            current_trip["end_time"] = trx_time
            current_trip["end_type"] = "GATE_OUT" if "GATE" in trx_type else "TOB_OUT"
            current_trip["end_terminal"] = transaction["terminal_name_var"]
            
            # Simpan trip yang sudah selesai
            trips.append(current_trip)
            current_trip = None  # Reset untuk trip berikutnya
    
    # Handle trip terakhir yang belum completed
    if current_trip:
        # Auto-complete jika sudah lewat batas waktu
        if not current_trip["is_completed"]:
            current_time = current_row["trx_on"]
            time_gap = (current_time - current_trip["start_time"]).total_seconds() / 3600
            if time_gap > MAX_TRIP_HOURS:
                current_trip["is_completed"] = True
        
        trips.append(current_trip)
    
    return trips


def detect_in_in_case(df, current_idx):
    """
    IMPROVED: Mendeteksi kasus IN-IN untuk handling UNBLOKIR yang benar.
    
    IN-IN case terjadi ketika ada dua transaksi IN berturut-turut tanpa OUT di antaranya.
    Dalam kasus ini, UNBLOKIR harus di-refund karena merupakan koreksi, bukan pembayaran perjalanan.
    
    Parameters:
        df (DataFrame): DataFrame semua transaksi  
        current_idx (int): Index transaksi UNBLOKIR yang sedang dianalisis
        
    Returns:
        dict or None: Info tentang IN-IN case jika ditemukan
    """
    try:
        current_row = df.loc[current_idx]
        current_card = current_row["card_number_var"]
        current_time = current_row["trx_on"]
        
        # Ambil transaksi sebelum UNBLOKIR untuk kartu yang sama
        previous_transactions = df[
            (df["card_number_var"] == current_card) &
            (df["trx_on"] < current_time) &
            (df.index < current_idx)
        ].sort_values(by="trx_on", ascending=False)
        
        if len(previous_transactions) < 2:
            return None
            
        # Cari pola IN-IN terakhir
        in_count = 0
        last_in_idx = None
        
        for idx, row in previous_transactions.iterrows():
            trx_type = row["trx"]
            
            if "[IN]" in trx_type and ("GATE" in trx_type or "TOB" in trx_type):
                in_count += 1
                if in_count == 1:
                    last_in_idx = idx
                elif in_count == 2:
                    # Found IN-IN pattern
                    return {
                        "is_in_in_case": True,
                        "first_in_idx": idx,
                        "second_in_idx": last_in_idx,
                        "first_in_time": row["trx_on"],
                        "second_in_time": previous_transactions.loc[last_in_idx]["trx_on"]
                    }
            elif "[OUT]" in trx_type or "UNBLOKIR" in trx_type:
                # Found OUT or UNBLOKIR, reset counter
                in_count = 0
                last_in_idx = None
                
        return None
        
    except Exception as e:
        return None


    """
    REVISI: Mencari perjalanan terakhir yang belum dibayar.
    UNBLOKIR bisa membayar trip ini.
    
    Logika unpaid trip:
    1. Trip dengan payments_count = 0 (tidak ada payment sama sekali)
    2. Trip yang incomplete (tidak ada OUT) dan sudah lewat waktu wajar (>1 jam)
    3. Trip yang hanya memiliki IN payment tapi tidak ada completion payment (OUT/UNBLOKIR)
    
    Parameters:
        df (DataFrame): DataFrame semua transaksi
        current_idx (int): Index transaksi yang sedang dianalisis
    
    Returns:
        dict or None: Info trip yang belum terbayar
    """
    trips = get_trips_for_card(df, current_idx)
    current_time = df.loc[current_idx]["trx_on"]
    
    # Cari dari trip terbaru ke lama
    for trip in reversed(trips):
        # Skip trip yang mengandung transaksi current
        contains_current = any(t["idx"] == current_idx for t in trip["transactions"])
        if contains_current:
            continue
        
        # Kondisi 1: Trip belum dibayar sama sekali
        if trip["payments_count"] == 0:
            return trip
        
        # Kondisi 2: Trip incomplete dan sudah lewat waktu wajar (> 1 jam)
        if not trip["is_completed"]:
            time_diff = (current_time - trip["start_time"]).total_seconds() / 3600
            if time_diff > 1.0:  # Lebih dari 1 jam
                return trip
        
        # Kondisi 3: Trip hanya ada IN payment, tidak ada completion payment (OUT/UNBLOKIR)
        # Cek apakah ada payment selain IN di trip ini
        has_completion_payment = False
        for trx in trip["transactions"]:
            if trx["is_payment"] and ("[OUT]" in trx["type"] or "UNBLOKIR" in trx["type"]):
                has_completion_payment = True
                break
        
        # Jika trip complete tapi hanya ada IN payment, masih dianggap perlu additional payment
        if trip["is_completed"] and not has_completion_payment and trip["payments_count"] > 0:
            return trip
    
    return None


def is_double_deduct(df, idx):
    """
    REVISI MAJOR: Logika deteksi DD sesuai dengan penjelasan yang benar.
    
    Logika DD:
    1. UNBLOKIR tanpa trip sebelumnya yang perlu dibayar = DD
    2. UNBLOKIR bersamaan waktu dengan IN berikutnya (< 5 menit, terminal sama) = DD  
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
        
        # Khusus untuk UNBLOKIR: selalu analisis DD meskipun bukan payment transaction
        if "UNBLOKIR" in row["trx"]:
            # UNBLOKIR bisa berupa deduction (positive fare) atau refund (negative fare)
            # Untuk analisis DD, kita gunakan absolute value dari fare
            fare_amount = abs(row["fare_int"]) if not pd.isnull(row["fare_int"]) else 0
            is_integration = is_integration_fare(fare_amount)
        else:
            # Untuk transaksi non-UNBLOKIR, hanya cek yang benar memotong saldo
            if not is_payment_transaction(row):
                return False, 0, "Bukan transaksi pembayaran", False
            fare_amount = row["fare_int"]
            is_integration = is_integration_fare(row["fare_int"])
        
        # === LOGIKA UNBLOKIR ===
        if "UNBLOKIR" in row["trx"]:
            current_card = row["card_number_var"]
            current_time = row["trx_on"]
            current_terminal = row["terminal_name_var"]
            
            # Bedakan antara UNBLOKIR deduction vs refund
            is_deduction = bool(row.get("deduct_boo", False))
            
            # PRIORITAS 1: Cari apakah ada trip sebelumnya yang belum dibayar
            unpaid_trip = find_last_unpaid_trip(df, idx)
            
            # PRIORITAS 1.5: Check for IN-IN case (IMPROVED handling)
            in_in_case = detect_in_in_case(df, idx)
            
            # PRIORITAS 2: Cek apakah UNBLOKIR bersamaan waktu dengan IN berikutnya (GAP TIMER)
            # Improved: Detect exact concurrent timing (same time) and near-concurrent
            next_in = df[
                (df["card_number_var"] == current_card) &
                (df["trx_on"] > current_time) &
                (df["trx_on"] <= current_time + pd.Timedelta(minutes=5)) &
                (df["trx"].str.contains(r"\[IN\]", na=False))
            ]
            
            # Also check for exactly concurrent transactions (same timestamp)
            concurrent_in = df[
                (df["card_number_var"] == current_card) &
                (df["trx_on"] == current_time) &
                (df.index > idx) &  # Only check transactions after current one
                (df["trx"].str.contains(r"\[IN\]", na=False))
            ]
            
            concurrent_with_next_in = False
            concurrent_reason = ""
            
            # Check for exactly concurrent transactions first (same timestamp)
            if not concurrent_in.empty:
                concurrent_in_row = concurrent_in.iloc[0]
                concurrent_terminal = concurrent_in_row["terminal_name_var"]
                
                # Same time, same terminal = definitely DD
                if is_same_terminal_group(current_terminal, concurrent_terminal):
                    concurrent_with_next_in = True
                    concurrent_reason = f"DD: UNBLOKIR bersamaan PERSIS dengan IN di terminal sama (gap 0s)"
                
            # If not exactly concurrent, check for near-concurrent (within 5 minutes)
            elif not next_in.empty:
                next_in_row = next_in.iloc[0]
                gap_seconds = (next_in_row["trx_on"] - current_time).total_seconds()
                next_terminal = next_in_row["terminal_name_var"]
                
                # Cek apakah di terminal yang sama (REVISI: terminal parsing)
                if is_same_terminal_group(current_terminal, next_terminal):
                    concurrent_with_next_in = True
                    concurrent_reason = f"DD: UNBLOKIR bersamaan dengan IN di terminal sama (gap {gap_seconds:.0f}s)"
            
            # Logika keputusan berdasarkan jenis UNBLOKIR:
            if is_deduction:
                # UNBLOKIR deduction: Prioritaskan concurrent timing rule
                if concurrent_with_next_in:
                    return True, fare_amount, concurrent_reason, is_integration
                elif in_in_case and in_in_case["is_in_in_case"]:
                    # IN-IN case: UNBLOKIR harus di-refund (DD)
                    return True, fare_amount, f"DD: UNBLOKIR dalam IN-IN case (idx {in_in_case['first_in_idx']}-{in_in_case['second_in_idx']})", is_integration
                elif unpaid_trip:
                    # Cek subsidi untuk deduction
                    if is_subsidi_time(unpaid_trip["start_time"]) and fare_amount == 3500:
                        return True, 1500, f"DD Subsidi: UNBLOKIR bayar trip subsidi idx {unpaid_trip['start_idx']}, seharusnya 2000", is_integration
                    else:
                        return False, 0, f"Sah: UNBLOKIR bayar trip idx {unpaid_trip['start_idx']}", is_integration
                else:
                    return True, fare_amount, "DD: UNBLOKIR tanpa trip yang perlu dibayar", is_integration
            else:
                # UNBLOKIR refund: Handle IN-IN case differently
                if in_in_case and in_in_case["is_in_in_case"]:
                    # IN-IN case: UNBLOKIR refund adalah koreksi yang sah
                    return False, 0, f"Sah: UNBLOKIR refund koreksi IN-IN case (idx {in_in_case['first_in_idx']}-{in_in_case['second_in_idx']})", is_integration
                elif unpaid_trip:
                    if is_subsidi_time(unpaid_trip["start_time"]) and fare_amount == 3500:
                        return True, 1500, f"DD Subsidi: UNBLOKIR bayar trip subsidi idx {unpaid_trip['start_idx']}, seharusnya 2000", is_integration
                    else:
                        return False, 0, f"Sah: UNBLOKIR refund untuk trip idx {unpaid_trip['start_idx']}", is_integration
                elif concurrent_with_next_in:
                    return True, fare_amount, concurrent_reason, is_integration
                else:
                    return True, fare_amount, "DD: UNBLOKIR refund tanpa justifikasi", is_integration
        
        # === LOGIKA IN/OUT dalam trip ===
        trips = get_trips_for_card(df, idx)
        current_trip = None
        payment_order_in_trip = 0
        
        # Cari trip yang mengandung transaksi ini
        for trip in trips:
            for i, transaction in enumerate(trip["transactions"]):
                if transaction["idx"] == idx:
                    current_trip = trip
                    # Hitung urutan payment dalam trip (IMPROVED: more robust counting)
                    payments_before = sum(1 for t in trip["transactions"][:i+1] if t["is_payment"])
                    payment_order_in_trip = payments_before
                    break
            if current_trip:
                break
        
        # IMPROVED: Stronger multiple payment detection
        if current_trip and payment_order_in_trip > 1:
            # Payment ke-2, ke-3, dst dalam trip yang sama = DD
            total_payments = sum(1 for t in current_trip["transactions"] if t["is_payment"])
            return True, fare_amount, f"DD: Payment ke-{payment_order_in_trip} dari {total_payments} dalam trip idx {current_trip['start_idx']}", is_integration
        
        # === CEK SUBSIDI untuk payment normal ===
        if is_subsidi_time(row["trx_on"]) and fare_amount == 3500:
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


def debug_card_trips(df, card_number, debug=True):
    """
    BARU: Debug fungsi untuk melihat trip detection untuk satu kartu tertentu.
    Berguna untuk testing dan troubleshooting.
    
    Parameters:
        df (DataFrame): DataFrame transaksi
        card_number (str): Nomor kartu yang ingin di-debug
        debug (bool): Print detail atau tidak
    
    Returns:
        list: List trips untuk kartu tersebut
    """
    card_data = df[df["card_number_var"] == card_number].sort_values("trx_on")
    
    if card_data.empty:
        if debug:
            print(f"Kartu {card_number} tidak ditemukan")
        return []
    
    if debug:
        print(f"\n=== DEBUG CARD {card_number} ===")
        print(f"Total transaksi: {len(card_data)}")
        
        for idx, row in card_data.iterrows():
            payment_status = "💰" if is_payment_transaction(row) else "⭕"
            print(f"idx {idx:2d}: {row['trx_on'].strftime('%H:%M:%S')} | {row['trx']:15s} | {row['terminal_name_var']:20s} | {payment_status}")
    
    # Analisis trips untuk transaksi terakhir kartu ini
    last_idx = card_data.index[-1]
    trips = get_trips_for_card(df, last_idx)
    
    if debug:
        print(f"\n=== TRIPS DETECTED ===")
        for i, trip in enumerate(trips, 1):
            status = "✅ Complete" if trip["is_completed"] else "🟡 Ongoing"
            paid_status = "💰 Paid" if trip["is_paid"] else "❌ Unpaid"
            
            print(f"\nTrip {i}: {status} | {paid_status}")
            print(f"  Start: idx {trip['start_idx']} | {trip['start_type']} | {trip.get('start_terminal', 'N/A')}")
            if trip.get("end_time"):
                print(f"  End:   {trip.get('end_type', 'N/A')} | {trip.get('end_terminal', 'N/A')}")
            print(f"  Payments: {trip['payments_count']}")
            
            for trx in trip["transactions"]:
                payment_mark = "💰" if trx["is_payment"] else "⭕"
                print(f"    - idx {trx['idx']:2d}: {trx['type']} | {payment_mark}")
    
    return trips
