#########################################
# app.py
#########################################

import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model


# ---------- STEP 1: Parsing / Data Extraction ----------

def parse_csv(file) -> pd.DataFrame:
    """
    Reads the uploaded CSV, returns a DataFrame with the necessary columns:
      - 'Profil'
      - 'Material'
      - 'Ant'
      - 'Lengde (stk) [mm]'
      - 'Lengde (SUM) [m]'
      - etc.
    """
    df = pd.read_csv(file, sep=None, engine='python')  # sep=None lets pandas guess
    return df


def extract_piece_data_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns:
      'Profil', 'Material', 'Ant', 'Lengde (stk) [mm]', ...
    Convert each row into repeated items (one row per required piece),
    storing:
      - row_id (which row in the original data)
      - profil
      - length_m (converted from mm)
    Return a 'long-form' DataFrame with one row per piece.
    """
    rows = []
    for idx, row in df.iterrows():
        profil = row.get('Profil', f'Unknown_{idx}')
        quantity = int(row.get('Ant', 1))  # how many pieces of this type
        length_mm = float(row.get('Lengde (stk) [mm]', 0))
        length_m = length_mm / 1000.0  # convert mm to m
        
        for _ in range(quantity):
            rows.append({
                'row_id': idx,
                'Profil': profil,
                'length_m': length_m
            })
    
    # Create a new DataFrame of individual pieces
    long_df = pd.DataFrame(rows)
    return long_df


def parse_manual_pieces(text_input: str) -> pd.DataFrame:
    """
    If the user didn't upload a CSV, this handles a text area with
    sub-lengths in meters (comma or newline separated).
    Returns a DataFrame with columns: ['row_id', 'Profil', 'length_m'].
    """
    pieces = []
    lines = [x.strip() for x in text_input.replace('\n', ',').split(',') if x.strip()]
    for i, val in enumerate(lines):
        try:
            length_m = float(val)
            pieces.append({
                'row_id': i,
                'Profil': f'Manual_{i+1}',
                'length_m': length_m
            })
        except ValueError:
            # if parsing fails, skip
            continue
    
    return pd.DataFrame(pieces)


# ---------- STEP 2: The Cutting Stock Solver (OR-Tools) ----------

def solve_cutting_stock(piece_df: pd.DataFrame, 
                        bar_lengths=(12, 15, 18), 
                        max_bars_per_length=10):
    """
    Solve a 1D cutting-stock / bin-packing style problem using OR-Tools CP-SAT.
    
    Inputs:
      piece_df: DataFrame with columns: ['row_id', 'Profil', 'length_m']
                Each row is one required piece.
      bar_lengths: tuple/list of available bar lengths in meters (default: 12, 15, 18)
      max_bars_per_length: how many bars of each length we allow as an upper bound.
    
    Returns a dictionary with:
      {
        'status': 'OPTIMAL' or something else,
        'objective': minimal total length used,
        'assignments': list of {
           'bar_type': one of the bar_lengths,
           'bar_index': int,
           'pieces': list of (piece_index, row_id, Profil, length_m),
           'used_length': float,
           'waste': float
        }
      }
    or None if infeasible.
    """
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    
    pieces = piece_df.to_dict('records')
    num_pieces = len(pieces)
    bar_types = list(bar_lengths)  # e.g. [12, 15, 18]
    
    # For indexing
    # We'll label bars by (t, b) where t is bar_types index, b in [0..max_bars_per_length-1]
    # x[i, t, b] = 1 if piece i is assigned to bar b of type t
    # used[t, b] = 1 if bar b of type t is used
    
    x = {}
    used = {}
    
    for t_idx, L in enumerate(bar_types):
        for b_idx in range(max_bars_per_length):
            used[(t_idx, b_idx)] = model.NewBoolVar(f"used_t{t_idx}_b{b_idx}")
            for i in range(num_pieces):
                x[(i, t_idx, b_idx)] = model.NewBoolVar(f"x_i{i}_t{t_idx}_b{b_idx}")
    
    # 1) Each piece assigned exactly once
    for i in range(num_pieces):
        model.Add(sum(x[(i, t_idx, b_idx)] 
                      for t_idx, L in enumerate(bar_types)
                      for b_idx in range(max_bars_per_length)) == 1)
    
    # 2) Capacity constraint: sum of lengths in each bar <= bar_length if used
    for t_idx, L in enumerate(bar_types):
        for b_idx in range(max_bars_per_length):
            model.Add(
                sum(pieces[i]['length_m'] * x[(i, t_idx, b_idx)]
                    for i in range(num_pieces)) 
                <= L * used[(t_idx, b_idx)]
            )
    
    # 3) If a piece is assigned to (t, b), that bar is considered used
    for t_idx, L in enumerate(bar_types):
        for b_idx in range(max_bars_per_length):
            for i in range(num_pieces):
                # x[i,t,b] <= used[t,b]
                model.Add(x[(i, t_idx, b_idx)] <= used[(t_idx, b_idx)])
    
    # Objective: minimize total bar length used
    # = sum(L * used[t,b] for all t,b)
    total_length = []
    for t_idx, L in enumerate(bar_types):
        for b_idx in range(max_bars_per_length):
            total_length.append(L * used[(t_idx, b_idx)])
    model.Minimize(sum(total_length))
    
    # Solve
    status = solver.Solve(model)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Build solution
        assignments = []
        for t_idx, L in enumerate(bar_types):
            for b_idx in range(max_bars_per_length):
                if solver.Value(used[(t_idx, b_idx)]) == 1:
                    # which pieces are assigned here?
                    assigned_pieces = []
                    for i in range(num_pieces):
                        if solver.Value(x[(i, t_idx, b_idx)]) == 1:
                            assigned_pieces.append(i)
                    
                    if assigned_pieces:
                        used_len = sum(pieces[i]['length_m'] for i in assigned_pieces)
                        waste = L - used_len
                        piece_details = [
                            (i, 
                             pieces[i]['row_id'], 
                             pieces[i]['Profil'], 
                             pieces[i]['length_m'])
                            for i in assigned_pieces
                        ]
                        assignments.append({
                            'bar_type': L,
                            'bar_index': b_idx,
                            'pieces': piece_details,
                            'used_length': used_len,
                            'waste': waste
                        })
        
        obj_value = sum([a['bar_type'] for a in assignments])  # since each bar_type is counted once per used bar
        result = {
            'status': 'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE',
            'objective': obj_value,
            'assignments': assignments
        }
        return result
    else:
        return None


# ---------- STEP 3: Streamlit App / UI ----------

def main():
    st.title("Cutting Stock Optimizer")
    st.write(
        """
        This app finds how many **12m, 15m, and 18m** bars are needed to fulfill
        all required pieces **with minimal waste**. You can:
        
        1. **Upload a CSV** with columns: Profil, Material, Ant, Lengde (stk) [mm], ...
        2. **Manually input** piece lengths (in meters) if you don't have a CSV.
        """
    )
    
    # -- File uploader
    uploaded_file = st.file_uploader("Upload CSV (with the specified columns)", type=["csv"])
    
    # -- Manual input
    st.subheader("Or input sub-lengths in meters manually (comma- or newline-separated):")
    manual_input = st.text_area("Enter lengths in m, e.g. 1.087, 1.088, 1.090...")
    
    # Container for processing
    piece_df = None
    
    if uploaded_file is not None:
        df = parse_csv(uploaded_file)
        st.write("Preview of uploaded CSV:")
        st.dataframe(df)
        
        # Convert to piece-level data
        piece_df = extract_piece_data_from_df(df)
        st.write(f"Total pieces extracted: {len(piece_df)}")
    
    elif manual_input.strip():
        # Use manual data
        piece_df = parse_manual_pieces(manual_input)
        st.write(f"Total pieces from manual input: {len(piece_df)}")
        if not piece_df.empty:
            st.dataframe(piece_df)
    
    # If we have data, show button to solve
    if piece_df is not None and not piece_df.empty:
        if st.button("Optimize Cutting Plan"):
            result = solve_cutting_stock(piece_df)
            
            if result is None:
                st.error("No feasible solution found.")
            else:
                st.success(f"Solution Status: {result['status']}")
                st.write(f"**Total Bar Length Used:** {result['objective']} m")
                
                total_requested_length = piece_df['length_m'].sum()
                waste = result['objective'] - total_requested_length
                st.write(f"**Total Required Length:** {total_requested_length:.3f} m")
                st.write(f"**Waste:** {waste:.3f} m")
                
                # Summarize usage by bar length
                bar_usage_count = {}
                for asg in result['assignments']:
                    bar_len = asg['bar_type']
                    bar_usage_count.setdefault(bar_len, 0)
                    bar_usage_count[bar_len] += 1
                st.write("**Bars Used (count by length)**:")
                for bar_len, count in bar_usage_count.items():
                    st.write(f"- {bar_len}m bars: {count}")
                
                # Detailed assignment table
                st.subheader("Detailed Cut Assignment")
                for asg in sorted(result['assignments'], key=lambda x: (x['bar_type'], x['bar_index'])):
                    bar_len = asg['bar_type']
                    b_idx = asg['bar_index']
                    used_len = asg['used_length']
                    wst = asg['waste']
                    
                    st.markdown(f"**Bar (Length={bar_len} m, Index={b_idx}):** used {used_len:.3f} m, waste {wst:.3f} m")
                    
                    # Show pieces in a small table
                    piece_table = pd.DataFrame(
                        [
                            {
                                "PieceIndex": p_i,
                                "OriginalRowID": row_id,
                                "Profil": profil,
                                "Length(m)": length_m
                            }
                            for (p_i, row_id, profil, length_m) in asg['pieces']
                        ]
                    )
                    st.table(piece_table)
    else:
        st.info("Upload a CSV or enter manual lengths to proceed.")


if __name__ == "__main__":
    main()