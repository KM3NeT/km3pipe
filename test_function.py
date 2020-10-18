def get_selected_items(point, item, d_min, d_max):

        item_pos = np.array([item.pos_x, item.pos_y, item.pos_z]).T
        distances = kp.math.dist(point, item_pos, axis=1)
        mask = (distances >= d_min) & (distances <= d_max)

        selected_items = item[mask]
  
        return selected_items
