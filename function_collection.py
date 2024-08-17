import os
# import math
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import re
import statsmodels.api as sm


import matplotlib as mpl
from matplotlib import pyplot as plt

def search_string_in_file(file_name, string_to_search):
    line_number = 0
    list_of_results = []
    with open(file_name, 'r') as read_file:
        for line in read_file:
            line_number += 1
            if string_to_search in line:
                list_of_results.append((line_number, line.rstrip()))
    return list_of_results

def read_cv_versastat(cv_file_path):
    # Search for line match beginning and end of CV data and give ln number
    start_segment = search_string_in_file(cv_file_path, 'Definition=Segment')[0][0]
    end_segment = search_string_in_file(cv_file_path, '</Segment')[0][0]
    # Count file total line number
    with open(cv_file_path, 'r') as file:
        ln_count = sum(1 for _ in file)
    with open(cv_file_path, 'r') as file:
        # Search for scan rate value
        # Search for the pattern using regex
        match = re.search(r'Scan Rate \(V/s\)=([\d.]+)', file.read())
        if match:
            # Extract the value from the matched pattern
            cv_file_scan_rate = float(match.group(1))
        else:
            cv_file_scan_rate = float(0)
    footer = ln_count-end_segment
    cv_df = pd.read_csv(cv_file_path, skiprows=start_segment, skipfooter=footer, usecols=[2,3], header=None, engine='python')
    cv_df = cv_df.dropna() #remove NaN
    cv_df.columns = [str(cv_file_path) +' volt', str(cv_file_path) +' current']
    return cv_df, cv_file_scan_rate

def read_cv_csv(cv_file_path):
    cv_df_single = pd.read_csv(cv_file_path,usecols=[0,1])
    cv_file_scan_rate = float(0)
    cv_file_scan_rate.append(cv_file_scan_rate)
    cv_df = cv_df_single.dropna() #remove NaN
    cv_df.columns = [str(cv_file_path) +' volt', str(cv_file_path) +' current']
    return cv_df, cv_file_scan_rate

def read_cv_text(cv_file_path):
    cv_df_single = pd.read_table(cv_file_path, sep='\t', header=None, usecols=[0,1])
    cv_file_scan_rate = float(0)
    cv_file_scan_rate.append(cv_file_scan_rate)
    cv_df = cv_df_single.dropna() #remove NaN
    cv_df.columns = [str(cv_file_path) +' volt', str(cv_file_path) +' current']
    return cv_df, cv_file_scan_rate

def read_cv_corrware(cv_file_path):
    start_segment = search_string_in_file(cv_file_path, 'End Comments')[0][0]
    with open(cv_file_path, 'r') as file:
        # Search for scan rate value
        # Search for the pattern using regex
        match = re.search(r'Scan Rate:\s+(\d+)', file.read())
        if match:
            # Extract the value from the matched pattern
            cv_file_scan_rate = float(match.group(1))
        else:
            cv_file_scan_rate = float(0)    
    footer = 0
    cv_df = pd.read_csv(cv_file_path,sep='\t',skiprows=start_segment, skipfooter=footer, usecols=[0,1], header=None, engine='python')
    cv_df = cv_df.dropna() #remove NaN
    cv_df.columns = [str(cv_file_path)+' volt', str(cv_file_path)+' current']
    return cv_df, cv_file_scan_rate

def read_cv_format(cv_file_path,cv_format):
    # Convert various cyclic voltammogram file format to pandas dataframe with the same format
    # cv_df: is the CV voltage and current
    # cv_param_df: is the basic setting (eg. scan rate) that can be found with the file,
    # but will later be modifiable by the user.
    # cv_2nd_deriv_concat_df: Perform 2nd derivative peak detection in CV, this should be modifiable by user
    # cv_file_path: path to the file
    # cv_format: specify format to read correctly
    cv_2nd_deriv_concat_df = pd.DataFrame()

    if cv_format == "CSV":
        cv_df, cv_file_scan_rate = read_cv_csv(cv_file_path)
    elif cv_format == "text":
        cv_df, cv_file_scan_rate = read_cv_text(cv_file_path)
    elif cv_format == "VersaSTAT":
        cv_df, cv_file_scan_rate = read_cv_versastat(cv_file_path)
    elif cv_format == "CorrWare":
        cv_df, cv_file_scan_rate = read_cv_corrware(cv_file_path)
    else:
        raise Exception("Unknown file type, please choose . cor, .csv, .par, .txt")

    blank_param = {
        'file path': [cv_file_path],
        'file name': [os.path.basename(cv_file_path)],
        'file format': [cv_format],
        'number of data points': [cv_df.shape[0]],
        'trim_start':0,
        'trim_end':[cv_df.shape[0]-1],
        'baseline_start_1':[0],
        'baseline_end_1':[0],
        'baseline_start_2':[0],
        'baseline_end_2':[0],
        'peak_pos_1':[0],
        'peak_pos_2':[0],
        'peak_range':[1],
        'peak_range_2':[0],
        'peak_mode_1':["max"],
        'peak_mode_2':["min"],
        'scan_rate':[cv_file_scan_rate],
        'elec_area':[1.0],
        'ir_compensation':[1.0],
        'nicholson_bool':[False],
        'jsp0':[0.0]}

    cv_param_df = pd.DataFrame(blank_param)
    idx_defl = peak_2nd_deriv(np.array(cv_df[cv_file_path+str(' volt')]),np.array(cv_df[cv_file_path+str(' current')]),0.05,0.05)
    cv_2nd_deriv_df = pd.DataFrame({'B':idx_defl})
    cv_2nd_deriv_df.columns = [cv_file_path]
    cv_2nd_deriv_concat_df = pd.concat([cv_2nd_deriv_concat_df,cv_2nd_deriv_df])
    return cv_df,cv_param_df,cv_2nd_deriv_concat_df

def battery_xls2df(bat_file):
    if bat_file.lower().endswith(".xls"):
        df_bat = pd.read_excel(bat_file,header=None)
        # Drop Index column, create our own
        df_bat = df_bat.drop([0],axis=1)
        # Delete all row that does not contain C_CC D_CC R
        row_size_raw_df_bat = len(df_bat)
        for i in range(0,row_size_raw_df_bat):
            bat_cell_state = pd.Series(df_bat[5])[i]
            if bat_cell_state != 'C_CC' and bat_cell_state != 'D_CC' and bat_cell_state != 'R' and bat_cell_state != 'C_CV' and bat_cell_state != 'D_CV':
                df_bat = df_bat.drop([i])
        df_bat.columns = ['time', 'volt', 'current', 'capacity', 'state']
        # Reset index after dropping some rows
        df_bat.reset_index(inplace=True)
        # convert '2-02:18:04' to seconds
        # Pandas datetime does not support changing format.
        row_size = len(df_bat)
        for i in range(0,row_size):
            df_bat['time'][i] = time2sec(df_bat['time'][i],'[:,-]')
            
        time_df = np.array(pd.Series(df_bat['time'])) #Get "time" column
        volt_df = np.array(pd.Series(df_bat['volt'].astype(float))) #Get "volt" column, convert to float
        current_df = np.array(pd.Series(df_bat['current'].astype(float))) #Get "Current" column, convert to float
        capacity_df = np.array(pd.Series(df_bat['capacity'].astype(float))) #Get "capacity" column, convert to float
        state_df = pd.Series(df_bat['state']) #Get "state" column
    else:
        raise Exception("Unknown file type, please choose .xls")
    return df_bat,row_size, time_df, volt_df, current_df, capacity_df, state_df

def open_battery_data(file_path,separate):
    # file_path string
    # separate string
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.xlsx' or file_extension == '.xls':
        df = pd.read_excel(file_path)
    elif file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension == '.txt':
        df = pd.read_csv(file_path, sep=separate, engine='python', header=None)
    elif file_extension == '.ods':
        df = pd.read_excel(file_path, engine='odf')
    else:
        print(f"Unsupported file format: {file_extension}. Please use .xlsx, .csv, .txt, .ods, or add a feature request")
        return None
    return df

def group_index(arr,key):
    arr = np.array(arr)
    state_ls = []
    for i in np.arange(0,len(arr),1):
      size = len(arr)
      if arr[i] == key:
        if i == 0 and arr[i+1] != key:
          state_ls.append(0,1)
        elif i == 0:
          state_ls.append(0)
        elif i > 0 and i < size-1:
          if arr[i-1] != key:
            state_ls.append(i)
          if arr[i+1] != key:
            state_ls.append(i+1)
        elif i == size-1:
            state_ls.append(i)
        elif i == size-1 and arr[i-1] != key:
          state_ls.append(size-1,size)
    state_group = np.array(state_ls).reshape(-1, 2)
    return state_group

def df_select_column(df,volt_col,current_col,time_col,rm_num_col):
    # Open all voltage, current, time header if not None
    df = df[[col for col in [volt_col,current_col,time_col] if col is not None]]
    if rm_num_col is not None:
    # Create a mask where True indicates the value is numeric
        df = df[df[rm_num_col].apply(lambda x: str(x).isnumeric())]
    df = df.reset_index(drop=True)
    return df    

def cut_list_to_shortest(a,b):
    min_length = min(len(a), len(b))
    a = a[:min_length]
    b = b[:min_length]
    return a,b

def search_pattern(lst, pattern):
    indices = []
    for i in range(len(lst)):
        if lst[i:i+len(pattern)] == pattern:
            indices.append(i)
    return indices

def get_CV_init(cv_df):
    volt = cv_df[:,0]
    current = cv_df[:,1]
    volt = volt[np.isfinite(volt)]
    current = current[np.isfinite(current)]
    cv_size = int(len(volt))
    return cv_size, volt, current

def ir_compen_func(volt,current,ir_compen):
    volt_compen = volt - current*ir_compen
    return volt_compen

def get_peak_CV(peak_mode, volt, current, peak_range, peak_pos, baseline,idx_defl):
    # peak_mode - string, "exact", "deflection", "min", "max"
    # cv_size - int
    # volt, current - array
    # peak_range - int
    # peak_pos - int
    # baseline - list where first element is where the baseline start, and second is where it end
    
    # low/high _range_peak - return index
    cv_size = len(volt)
    baseline = list(baseline)
    baseline.sort()
    jp_lns = baseline[0]
    jp_lne = baseline[1]
    # If peak range is given as 0, then peak is just where peak position is
    if peak_mode == "exact":
        peak_curr = current[peak_pos] # ###########NEED TO DEPEND ON PEAK_POS
        peak_volt = volt[peak_pos]
        low_range_peak = peak_pos
        high_range_peak = peak_pos 
        peak_range = 0
    elif peak_mode == "2nd derivative":            
        low_range_peak = peak_pos
        high_range_peak = peak_pos    
        peak_curr = current[idx_defl]
        peak_volt = volt[idx_defl]      
        peak_range = 0
    # Search for peak between peak_range.     
    elif peak_mode == "max" or peak_mode == "min":
        if peak_range == 0:
            peak_range = 1
        high_range_peak = np.where((peak_pos+peak_range)>=(cv_size-1),(cv_size-1),peak_pos+peak_range)
        low_range_peak = np.where((peak_pos-peak_range)>=0,peak_pos-peak_range,0)
        peak_curr_range = current[low_range_peak:high_range_peak]    
        if peak_mode == "max":
            peak_curr = max(peak_curr_range)          
        elif peak_mode == "min":
            peak_curr = min(peak_curr_range)
            
        peak_idx = np.argmin(np.abs(peak_curr_range-peak_curr))     
        peak_volt = volt[low_range_peak:high_range_peak][peak_idx]
           
    # If the extrapolation coordinate overlapped, just give horizontal line
    if (volt[jp_lns:jp_lne]).size == 0:
        volt_jp = np.array([0, 1])
        current_jp = np.array([0, 0])
    else:
        volt_jp = volt[jp_lns:jp_lne]
        current_jp = current[jp_lns:jp_lne]
        
    jp_lnfit_coef,_ = poly.polyfit(volt_jp,current_jp, 1, full=True) # 1 for linear fit  
    jp_poly1d = poly.Polynomial(jp_lnfit_coef) 
    jp = peak_curr - jp_poly1d(peak_volt)
    return low_range_peak, high_range_peak, peak_volt, peak_curr, jp, jp_poly1d

def linear_fit(volt, current):
    fit_coef,_ = poly.polyfit(volt,current, 1, full=True) # 1 for linear fit  
    poly1d = poly.Polynomial(fit_coef) 
    return fit_coef, poly1d 

def time2sec(time_raw,delim):
    # Take time format such as 1-12:05:24 and convert to seconds
    time_raw = str(time_raw)
    time_sp = re.split(delim, time_raw)
    time_sp = list(map(int, time_sp))
    if len(time_sp) == 4:
        time_sec = time_sp[0]*3600*24 + time_sp[1]*3600 + time_sp[2]*60 + time_sp[3]
    elif len(time_sp) == 3:
        time_sec = time_sp[0]*3600 + time_sp[1]*60 + time_sp[2]
    return int(time_sec)

# def find_state_seq(state_df):
#     charge_CC_seq = find_seg_start_end(state_df,'C_CC')
#     discharge_CC_seq = find_seg_start_end(state_df,'D_CC')
#     charge_CV_seq = find_seg_start_end(state_df,'C_CV')
#     discharge_CV_seq = find_seg_start_end(state_df,'D_CV')
#     rest_seq = find_seg_start_end(state_df,'R')
#     return charge_CC_seq, discharge_CC_seq, rest_seq, charge_CV_seq, discharge_CV_seq

def get_battery_eff(row_size, time_df, volt_df, current_df, capacity_df, state_df, charge_seq, discharge_seq):
    # Calculate the area of charge and discharge cycle and find VE,CE,EE for each cycle
    VE_lst = []
    CE_lst = []
    charge_cap_lst = []
    discharge_cap_lst = []
    cycle_end = min(np.shape(charge_seq)[0],np.shape(discharge_seq)[0]) #take the min amount of cycle between the charge and dis
    # cycle_start = 1
    for i in range(0,cycle_end):
        # Error if the cycle is not complete charge sequence more than discharge sequence
        time_seq_C = time_df[charge_seq[i][0]:charge_seq[i][1]+1]
        volt_seq_C = volt_df[charge_seq[i][0]:charge_seq[i][1]+1]
        current_seq_C = current_df[charge_seq[i][0]:charge_seq[i][1]+1]
        charge_cap_seq_C = capacity_df[charge_seq[i][0]:charge_seq[i][1]+1] 
        
        time_seq_D = time_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
        volt_seq_D = volt_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
        current_seq_D = current_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
        dis_cap_seq_C = capacity_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
        
        int_vt_C = np.trapz(volt_seq_C,time_seq_C)
        int_vt_D = np.trapz(volt_seq_D,time_seq_D)
        int_ct_C = np.trapz(current_seq_C,time_seq_C)
        # During discharge, current is negative, must make to positive
        int_ct_D = -(np.trapz(current_seq_D,time_seq_D))
        VE = int_vt_D/int_vt_C
        CE = int_ct_D/int_ct_C

        charge_cap_seq = np.array(charge_cap_seq_C)[-1]
        dis_cap_seq = np.array(dis_cap_seq_C)[-1]

        charge_cap_lst.append(charge_cap_seq)
        discharge_cap_lst.append(dis_cap_seq)
      
        VE_lst.append(VE)
        CE_lst.append(CE)
    VE_arr = np.array(VE_lst) * 100 # convert to %
    CE_arr = np.array(CE_lst) * 100
    EE_arr = (VE_arr/100 * CE_arr/100)*100
    charge_cap_arr = np.array(charge_cap_lst)
    discharge_cap_arr = np.array(discharge_cap_lst)
    return VE_arr, CE_arr, EE_arr, charge_cap_arr, discharge_cap_arr, cycle_end

def cy_idx_state_range(state_df, cycle_start, cycle_end, charge_seq, discharge_seq):
    # Get index for beginning and end of specify cycle
    # Take all start and end of the cycle chosen, select the first and last.
    # For plotting purpose
    cycle_index = np.stack((charge_seq[cycle_start:cycle_end], discharge_seq[cycle_start:cycle_end])) #no need to include rest
    cycle_idx_start = np.amin(cycle_index)
    cycle_idx_end = np.amax(cycle_index)
    cycle_idx_range = [cycle_idx_start, cycle_idx_end]
    return cycle_idx_range

def lowess_func(x,y,frac):
    lowess = sm.nonparametric.lowess(y, x, frac=frac)
    smh_x = lowess[:, 0]
    smh_y = lowess[:, 1]
    return smh_x, smh_y

def diff(x,y):
    diff_y = np.gradient(y,x)
    # Find indices where x is not NaN
    valid_indices = np.isfinite(diff_y)
    # Use boolean indexing to select values in y corresponding to valid x values
    x = x[valid_indices]
    diff_y = diff_y[valid_indices]
    return x, diff_y #This is y

def lowess_diff(x_idx,x,y,frac):
    _, smh_y = lowess_func(x_idx,y,frac)
    x, smh_diff_y = diff(x,smh_y)
    return x, smh_diff_y

def peak_2nd_deriv(volt,current,frac_1=0.05,frac_2=0.05):
    # frac smoothness value, try with 0.05
    # The idx_arr is use to "unwarp" the circular CV
    idx_arr = np.arange(0,len(volt))
    _,smh_curr = lowess_func(idx_arr,current,frac_1)
    _,smh_volt = lowess_func(idx_arr,volt,frac_1)
 
    smh_volt,diff1_curr = diff(smh_volt,smh_curr) #First diff, find peaks (slope = 0)

    idx_arr = np.arange(0,len(smh_volt)) # Recalculate size of idx_arr because NaN and inf removed
    smh_volt,diff2_curr = lowess_diff(idx_arr,smh_volt,diff1_curr,frac_2)
    smh_volt,diff3_curr = lowess_diff(idx_arr,smh_volt,diff2_curr,0) #Detect deflection

    idx_intc_defl = idx_intercept(0,diff3_curr)
    idx_intc_defl = [int(x) for x in idx_intc_defl]
    return idx_intc_defl 

def idx_intercept(yint,y):
    idx_intc = []
    y = np.squeeze(y)
    for i in np.arange(1,y.size):
        if y[i] == yint:
            idx_intc.append(i)
        elif y[i] < yint and y[i-1] > yint or y[i] > yint and y[i-1] < yint: #in negative
            new_x = np.interp(yint, y[i-1:i+1], [i-1,i])
            idx_intc.append(new_x)
    return list(idx_intc)

def diffusion(scan,jp,alpha,conc_bulk,n):
# For more info - Electrochemical Methods: Fundamentals and Applications, 3rd Edition Allen J. Bard, Larry R. Faulkner, Henry S. White
# - Redox Flow Batteries: How to Determine Electrochemical Kinetic Parameters, Hao Wang et al.
# scan_rate_arr - scan rate,unit in volt
# jp - peak current density, unit in A/cm2
# alpha - charge-transfer coefficient, no unit
# conc_bulk - Bulk concentration, unit in mol/cm3
# n - number of electrons, no unit
    jp = np.abs(jp) 
    sqrt_scan = np.sqrt(scan)
    try: 
        jp_arr_lnfit, _ = poly.polyfit(sqrt_scan,jp,1,full=True)
        jp_arr_poly = poly.Polynomial(jp_arr_lnfit)
        jp_slope = jp_arr_lnfit[1] # take slope
        D_rev = (jp_slope/(2.69*(10**5)*n**(3/2)*conc_bulk))**2 # reversible
        D_irr = (jp_slope/(2.99*(10**5)*n**(3/2)*(alpha**0.5)*conc_bulk))**2 # irreversible  
    except SystemError:
        pass  
    # Calculate R2
    jp_fit = jp_arr_poly(sqrt_scan)
    residuals = jp - jp_fit
    ssr = np.sum(residuals ** 2)
    sst = np.sum((jp - np.mean(jp)) ** 2)
    r2 = (1 - (ssr / sst))
    return sqrt_scan, jp_fit ,D_irr ,D_rev ,r2

def reaction_rate(e_e0,jp,conc_bulk,n):
    jp = np.abs(jp)
    lnjp = np.log(jp)
    try:     
        lnjp_lnfit, _ = poly.polyfit(e_e0,lnjp,1,full=True)
        lnjp_poly = poly.Polynomial(lnjp_lnfit)
        lnjp_b = lnjp_lnfit[0] # take intercept
        slope = lnjp_lnfit[1]
        F = 96485.332
        alpha_cat = -slope*8.314472*298.15/F #cathodic where slope is negative
        alpha_ano = 1 + alpha_cat #anodic where slope is positive
        k0 = np.exp(lnjp_b-np.log(0.227*F*n*conc_bulk))
    except SystemError:
        pass
    # Calculate R2
    lnjp_fit = lnjp_poly(e_e0)       
    residuals = lnjp - lnjp_fit
    ssr = np.sum(residuals ** 2)
    sst = np.sum((lnjp - np.mean(lnjp)) ** 2)
    r2 = (1 - (ssr / sst))
    return lnjp, lnjp_fit, k0, alpha_cat, alpha_ano, r2

def find_alpha(volt,curr,baseline,peak_pos,jp_poly1d,jp,peak_volt):
    baseline = list(baseline)
    baseline.sort()
    jp_lns = baseline[0]
    volt_eval_jp = volt[jp_lns:peak_pos]
    curr_eval_jp = curr[jp_lns:peak_pos]
    try: 
        baseline_eval_jp = np.linspace(jp_poly1d(volt[jp_lns]),jp_poly1d(volt[peak_pos]),volt_eval_jp.size)
        curr_baseline_jp = curr_eval_jp-baseline_eval_jp
        ep12_jp_idx = (np.abs(curr_baseline_jp-jp/2)).argmin()
        ep12_jp = volt_eval_jp[ep12_jp_idx] #Potential at peak current 1/2 (Ep 1/2)
        jp12_jp = curr_eval_jp[ep12_jp_idx]
        alpha_jp = 1-((47.7/1000)/np.abs(peak_volt - ep12_jp))
    except (ValueError, IndexError):
        ep12_jp = 0
        jp12_jp = 0
        alpha_jp = 0
    return ep12_jp, jp12_jp, alpha_jp

def convert_ref_elec():
    ref_she = 0
    ref_sce_sat = 0.241 #Saturated calomel electrode
    ref_cse = 0.314
    ref_agcl_sat = 0.197 # saturated
    ref_agcl_3molkg = 0.210 # 3 mol KCl/kg
    ref_agcl_3moll = 0. # 3.0 mol KCl/L
    ref_hg2so4_sat = 0.64 # saturated k2so4
    ref_hg2so4_05 = 0.68 # 0.5 M H2SO4
    return 0
    
def min_max_peak(peak_mode,cv_size, volt, current, peak_range, peak_pos):
    high_range_peak = np.where((peak_pos+peak_range)>=(cv_size-1),(cv_size-1),peak_pos+peak_range)
    low_range_peak = np.where((peak_pos-peak_range)>=0,peak_pos-peak_range,0)
    peak_curr_range = current[low_range_peak:high_range_peak]
    
    if peak_mode == 'max':
        peak_curr = max(peak_curr_range)
        peak_idx = np.argmin(np.abs(peak_curr_range-peak_curr))
        peak_volt = volt[low_range_peak:high_range_peak][peak_idx]
    elif peak_mode == 'min':
        peak_curr = min(peak_curr_range)
        peak_idx = np.argmin(np.abs(peak_curr_range-peak_curr))
        peak_volt = volt[low_range_peak:high_range_peak][peak_idx]
    elif peak_mode == 'none':
        peak_curr = current[peak_pos]
        peak_volt = volt[peak_pos]
    peak_real_idx = int(peak_pos-peak_range+peak_idx)
    return high_range_peak, low_range_peak, peak_volt, peak_curr, peak_real_idx

def check_val(val, val_type, err_val):
    if val_type == "int":
        try:
            value = int(val)
        except ValueError:
            value = int(err_val)
    elif val_type == "float":
        try:
            value = float(val)
        except ValueError:
            value = float(err_val)
    return value

def switch_val(a,b):
    if a >= b:
        b_old = b
        b = a
        a = b_old
    if a == b: #Prevent overlapped
        b = a+1
    return a,b

def RDE_kou_lev(ror,lim_curr,conc_bulk,n,kinvis,ror_unit_arr):
    unit_mapping = {'RPM': 0.104719755,'rad/s': 1}
    conv_unit_arr = [unit_mapping.get(item, item) for item in ror_unit_arr] #Convert RPM to rad/s
    ror = ror * conv_unit_arr
    inv_sqrt_ror = 1/np.sqrt(ror)
    inv_lim_curr = 1/lim_curr 
    try:
        if kinvis <= 0:
            kinvis = np.NaN
        j_inv_lnfit, _ = poly.polyfit(inv_sqrt_ror,inv_lim_curr,1,full=True)
        kou_lev_polyfit = poly.Polynomial(j_inv_lnfit)
        j_kin = 1/j_inv_lnfit[0]
        
        slope = 1/j_inv_lnfit[1]
        F = 96485.332 #Faraday constant
        # Levich equation
        diffusion = (slope/(0.62*n*F*kinvis**(-1/6)*conc_bulk))**(3/2) #cathodic where slope is negative
        # Calculate R2
        j_inv_fit = kou_lev_polyfit(inv_sqrt_ror)   
        residuals = inv_lim_curr - j_inv_fit
        ssr = np.sum(residuals ** 2)
        sst = np.sum((inv_lim_curr - np.mean(inv_lim_curr)) ** 2)
        r2 = (1 - (ssr / sst))
    except SystemError:
        j_kin=0
        r2=0
        j_inv_fit = []
        inv_sqrt_ror = []
        diffusion = np.NaN
        kou_lev_polyfit = np.NaN
    return inv_sqrt_ror, j_inv_fit, diffusion, j_kin, kou_lev_polyfit, r2

def data_poly_inter(x,y,poly_coef,detail):
    #Find the value of intersection
    # x,y is an array of equal length
    # poly_coef is 2nd data as numpy poly.Polynomial()
    #detail is number of points
    abs_min = np.abs(y-poly_coef(x))
    min_idx = np.argmin(abs_min)
    if min_idx == len(x):
        xvals1 = np.linspace(x[min_idx-1], x[min_idx],detail)
        yinterp1 = np.interp(xvals1, x, y)
        yinterp2 = np.interp(xvals1, x, poly_coef(x))
        fine_min_idx = np.argmin(np.abs(yinterp1-yinterp2))
    elif min_idx == 0:
        xvals1 = np.linspace(x[min_idx],x[min_idx+1],detail)
        yinterp1 = np.interp(xvals1, x, y)
        yinterp2 = np.interp(xvals1, x, poly_coef(x))
        fine_min_idx = np.argmin(np.abs(yinterp1-yinterp2))              
    elif abs_min[min_idx+1] > abs_min[min_idx-1]:
        xvals1 = np.linspace(x[min_idx-1], x[min_idx],detail)
        yinterp1 = np.interp(xvals1, x, y)
        yinterp2 = np.interp(xvals1, x, poly_coef(x))
        fine_min_idx = np.argmin(np.abs(yinterp1-yinterp2))
    elif abs_min[min_idx+1] < abs_min[min_idx-1]:
        xvals1 = np.linspace(x[min_idx],x[min_idx+1],detail)
        yinterp1 = np.interp(xvals1, x, y)
        yinterp2 = np.interp(xvals1, x, poly_coef(x))
        fine_min_idx = np.argmin(np.abs(yinterp1-yinterp2)) 
    elif abs_min[min_idx+1] == abs_min[min_idx-1]:
        fine_min_idx = min_idx
        xvals1 = x
        yinterp1 = y
    return xvals1[fine_min_idx], yinterp1[fine_min_idx]  