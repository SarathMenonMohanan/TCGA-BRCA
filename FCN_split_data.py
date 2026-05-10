
def FCN_split_data(d_frame,y_col = 'vital.status'):
    x = d_frame.drop(columns = [y_col])
    y = d_frame[y_col]
    return x,y