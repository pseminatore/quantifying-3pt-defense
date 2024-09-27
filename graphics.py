from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from data import read_tracking, read_locations, find_best_frame, get_play_df
import pandas as pd
import numpy as np
import xgboost as xgb
#from features import get_shooter_velocity

def get_feature_importance(model):
	xgb.plot_importance(model)
	plt.rcParams['figure.figsize'] = [5, 5]
	plt.show() 
	
	
def plot_obfuscation_scores(df):
	plt.scatter(df['court_x_d'], df['court_y_d'], s=100, c=df['obfuscation_score'])
	plt.xlim(0, 100)
	plt.ylim(-26, 26)
	plt.show()
		




def create_ncaa_full_court(ax=None, three_line='mens', court_color='#dfbb85',
						   lw=3, lines_color='black', lines_alpha=0.5,
						   paint_fill='blue', paint_alpha=0.4,
						   inner_arc=False, margin=2):
	"""
	Version 2020.2.19
	Creates NCAA Basketball Court
	Dimensions are in feet (Court is 97x50 ft)
	Created by: Rob Mulla / https://github.com/RobMulla

	* Note that this function uses "feet" as the unit of measure.
	* NCAA Data is provided on a x range: 0, 100 and y-range 0 to 100
	* To plot X/Y positions first convert to feet like this:
	```
	Events['X_'] = (Events['X'] * (94/100))
	Events['Y_'] = (Events['Y'] * (50/100))
	```
	
	ax: matplotlib axes if None gets current axes using `plt.gca`


	three_line: 'mens', 'womens' or 'both' defines 3 point line plotted
	court_color : (hex) Color of the court
	lw : line width
	lines_color : Color of the lines
	lines_alpha : transparency of lines
	paint_fill : Color inside the paint
	paint_alpha : transparency of the "paint"
	inner_arc : paint the dotted inner arc
	"""
	if ax is None:
		ax = plt.gca()

	# Create Pathes for Court Lines
	center_circle = Circle((94/2, 50/2), 6,
						   linewidth=lw, color=lines_color, lw=lw,
						   fill=False, alpha=lines_alpha)
	hoop_left = Circle((5.25, 50/2), 1.5 / 2,
					   linewidth=lw, color=lines_color, lw=lw,
					   fill=False, alpha=lines_alpha)
	hoop_right = Circle((94-5.25, 50/2), 1.5 / 2,
						linewidth=lw, color=lines_color, lw=lw,
						fill=False, alpha=lines_alpha)

	# Paint - 18 Feet 10 inches which converts to 18.833333 feet - gross!
	left_paint = Rectangle((0, (50/2)-6), 18.833333, 12,
						   fill=paint_fill, alpha=paint_alpha,
						   lw=lw, edgecolor=None)
	right_paint = Rectangle((94-18.83333, (50/2)-6), 18.833333,
							12, fill=paint_fill, alpha=paint_alpha,
							lw=lw, edgecolor=None)
	
	left_paint_boarder = Rectangle((0, (50/2)-6), 18.833333, 12,
						   fill=False, alpha=lines_alpha,
						   lw=lw, edgecolor=lines_color)
	right_paint_boarder = Rectangle((94-18.83333, (50/2)-6), 18.833333,
							12, fill=False, alpha=lines_alpha,
							lw=lw, edgecolor=lines_color)

	left_arc = Arc((18.833333, 50/2), 12, 12, theta1=-
				   90, theta2=90, color=lines_color, lw=lw,
				   alpha=lines_alpha)
	right_arc = Arc((94-18.833333, 50/2), 12, 12, theta1=90,
					theta2=-90, color=lines_color, lw=lw,
					alpha=lines_alpha)
	
	leftblock1 = Rectangle((7, (50/2)-6-0.666), 1, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	leftblock2 = Rectangle((7, (50/2)+6), 1, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	ax.add_patch(leftblock1)
	ax.add_patch(leftblock2)
	
	left_l1 = Rectangle((11, (50/2)-6-0.666), 0.166, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	left_l2 = Rectangle((14, (50/2)-6-0.666), 0.166, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	left_l3 = Rectangle((17, (50/2)-6-0.666), 0.166, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	ax.add_patch(left_l1)
	ax.add_patch(left_l2)
	ax.add_patch(left_l3)
	left_l4 = Rectangle((11, (50/2)+6), 0.166, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	left_l5 = Rectangle((14, (50/2)+6), 0.166, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	left_l6 = Rectangle((17, (50/2)+6), 0.166, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	ax.add_patch(left_l4)
	ax.add_patch(left_l5)
	ax.add_patch(left_l6)
	
	rightblock1 = Rectangle((94-7-1, (50/2)-6-0.666), 1, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	rightblock2 = Rectangle((94-7-1, (50/2)+6), 1, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	ax.add_patch(rightblock1)
	ax.add_patch(rightblock2)

	right_l1 = Rectangle((94-11, (50/2)-6-0.666), 0.166, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	right_l2 = Rectangle((94-14, (50/2)-6-0.666), 0.166, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	right_l3 = Rectangle((94-17, (50/2)-6-0.666), 0.166, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	ax.add_patch(right_l1)
	ax.add_patch(right_l2)
	ax.add_patch(right_l3)
	right_l4 = Rectangle((94-11, (50/2)+6), 0.166, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	right_l5 = Rectangle((94-14, (50/2)+6), 0.166, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	right_l6 = Rectangle((94-17, (50/2)+6), 0.166, 0.666,
						   fill=True, alpha=lines_alpha,
						   lw=0, edgecolor=lines_color,
						   facecolor=lines_color)
	ax.add_patch(right_l4)
	ax.add_patch(right_l5)
	ax.add_patch(right_l6)
	
	# 3 Point Line
	if (three_line == 'mens') | (three_line == 'both'):
		# 22' 1.75" distance to center of hoop
		three_pt_left = Arc((6.25, 50/2), 44.291, 44.291, theta1=-78,
							theta2=78, color=lines_color, lw=lw,
							alpha=lines_alpha)
		three_pt_right = Arc((94-6.25, 50/2), 44.291, 44.291,
							 theta1=180-78, theta2=180+78,
							 color=lines_color, lw=lw, alpha=lines_alpha)

		# 4.25 feet max to sideline for mens
		ax.plot((0, 11.25), (3.34, 3.34),
				color=lines_color, lw=lw, alpha=lines_alpha)
		ax.plot((0, 11.25), (50-3.34, 50-3.34),
				color=lines_color, lw=lw, alpha=lines_alpha)
		ax.plot((94-11.25, 94), (3.34, 3.34),
				color=lines_color, lw=lw, alpha=lines_alpha)
		ax.plot((94-11.25, 94), (50-3.34, 50-3.34),
				color=lines_color, lw=lw, alpha=lines_alpha)
		ax.add_patch(three_pt_left)
		ax.add_patch(three_pt_right)

	if (three_line == 'womens') | (three_line == 'both'):
		# womens 3
		three_pt_left_w = Arc((6.25, 50/2), 20.75 * 2, 20.75 * 2, theta1=-85,
							  theta2=85, color=lines_color, lw=lw, alpha=lines_alpha)
		three_pt_right_w = Arc((94-6.25, 50/2), 20.75 * 2, 20.75 * 2,
							   theta1=180-85, theta2=180+85,
							   color=lines_color, lw=lw, alpha=lines_alpha)

		# 4.25 inches max to sideline for mens
		ax.plot((0, 8.3), (4.25, 4.25), color=lines_color,
				lw=lw, alpha=lines_alpha)
		ax.plot((0, 8.3), (50-4.25, 50-4.25),
				color=lines_color, lw=lw, alpha=lines_alpha)
		ax.plot((94-8.3, 94), (4.25, 4.25),
				color=lines_color, lw=lw, alpha=lines_alpha)
		ax.plot((94-8.3, 94), (50-4.25, 50-4.25),
				color=lines_color, lw=lw, alpha=lines_alpha)

		ax.add_patch(three_pt_left_w)
		ax.add_patch(three_pt_right_w)

	# Add Patches
	ax.add_patch(left_paint)
	ax.add_patch(left_paint_boarder)
	ax.add_patch(right_paint)
	ax.add_patch(right_paint_boarder)
	ax.add_patch(center_circle)
	ax.add_patch(hoop_left)
	ax.add_patch(hoop_right)
	ax.add_patch(left_arc)
	ax.add_patch(right_arc)
	
	if inner_arc:
		left_inner_arc = Arc((18.833333, 50/2), 12, 12, theta1=90,
							 theta2=-90, color=lines_color, lw=lw,
					   alpha=lines_alpha, ls='--')
		right_inner_arc = Arc((94-18.833333, 50/2), 12, 12, theta1=-90,
						theta2=90, color=lines_color, lw=lw,
						alpha=lines_alpha, ls='--')
		ax.add_patch(left_inner_arc)
		ax.add_patch(right_inner_arc)

	# Restricted Area Marker
	restricted_left = Arc((6.25, 50/2), 8, 8, theta1=-90,
						theta2=90, color=lines_color, lw=lw,
						alpha=lines_alpha)
	restricted_right = Arc((94-6.25, 50/2), 8, 8,
						 theta1=180-90, theta2=180+90,
						 color=lines_color, lw=lw, alpha=lines_alpha)
	ax.add_patch(restricted_left)
	ax.add_patch(restricted_right)
	
	# Backboards
	ax.plot((4, 4), ((50/2) - 3, (50/2) + 3),
			color=lines_color, lw=lw*1.5, alpha=lines_alpha)
	ax.plot((94-4, 94-4), ((50/2) - 3, (50/2) + 3),
			color=lines_color, lw=lw*1.5, alpha=lines_alpha)
	ax.plot((4, 4.6), (50/2, 50/2), color=lines_color,
			lw=lw, alpha=lines_alpha)
	ax.plot((94-4, 94-4.6), (50/2, 50/2),
			color=lines_color, lw=lw, alpha=lines_alpha)

	# Half Court Line
	ax.axvline(94/2, color=lines_color, lw=lw, alpha=lines_alpha)

	# Boarder
	boarder = Rectangle((0.3,0.3), 94-0.4, 50-0.4, fill=False, lw=3, color='black', alpha=lines_alpha)
	ax.add_patch(boarder)
	
	# Plot Limit
	ax.set_xlim(-margin, 94 + margin)  # Court length is 94 feet
	ax.set_ylim(-margin, 50 + margin)
	ax.set_facecolor(court_color)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_xlabel('')
	return ax


def show_frame(frame_num,df_offense,df_shooter,ax,teammates,shot):
		closest_positions_offense = df_offense[df_offense['frame'] == frame_num]
		closest_positions_shooter = df_shooter[df_shooter['frame'] == frame_num]

				
				# Plot positions of offensive players
		ax.plot(closest_positions_offense['x_smooth'], closest_positions_offense['y_smooth'], 
						'o', markerfacecolor='#b94b75', markeredgecolor='black', markersize=10, linestyle='None', label='Offense')

				# Plot position of the shooter
		ax.plot(closest_positions_shooter['x_smooth'], closest_positions_shooter['y_smooth'], 
						'o', markerfacecolor='yellow', markeredgecolor='black', markersize=12, linestyle='None', label='Shooter')


		for _, row in teammates.iterrows():
				ax.plot(row['court_x'], row['court_y'], 
				marker='o', color='blue', markersize=8, linestyle='None', label=f"Teammate {row['annotation_code']}")

		if not shot.empty:
				ax.plot(shot['court_x'].values[0], shot['court_y'].values[0], 
						marker='x', color='red', markersize=12, linestyle='None', label='Shot Location')
				# Optionally add legends and other details

		speed, angle,start_pos,end_pos = get_shooter_velocity(frame_num=frame_num, df_tracking=df_shooter,frames_bef=20,frames_aft=20)
		
		vx = end_pos[0] - start_pos[0]
		vy = end_pos[1] - start_pos[1]

# Normalize to match the speed
		magnitude_velocity = np.sqrt(vx**2 + vy**2)
		vx = (vx / magnitude_velocity) * speed
		vy = (vy / magnitude_velocity) * speed

# Plot the velocity vector at the shooter's end position
		ax.quiver(end_pos[0], end_pos[1], vx, vy, angles='xy', scale_units='xy', scale=1, color='red', label=f"Velocity Vector\nSpeed: {speed:.2f}\nAngle: {angle:.2f}°")
		ax.legend(loc='upper right')
		plt.show()
		return
	 

def animate_play(game_id, play_id,example=False,smooth=True, shot_loc=True,show_closest_frame=False):
	if example == False:
		df_offense, df_defense, df_shooter = get_play_df(game_id, play_id)
	else:
		df_play =  pd.read_csv('example_play.csv')
		df_play['play_iid'] = df_play['game_id'].astype(str) + '-' + df_play['play_id'].astype(str)
		
		df_offense = df_play[df_play['type'] == 'teammate']
		df_defense = df_play[df_play['type'] == 'defender']
		df_shooter = df_play[df_play['type'] == 'shooter']


	
	if smooth == True:
		window = 9
	else: 
		window = 1

	df_offense['x_smooth'] = df_offense['x'].rolling(window=window).mean()
	df_offense['y_smooth'] = df_offense['y'].rolling(window=window).mean()

	df_defense['x_smooth'] = df_defense['x'].rolling(window=window).mean()
	df_defense['y_smooth'] = df_defense['y'].rolling(window=window).mean()

	df_shooter['x_smooth'] = df_shooter['x'].rolling(window=window).mean()
	df_shooter['y_smooth'] = df_shooter['y'].rolling(window=window).mean()


		#- 1 - 0.05 * df_offense['x_smooth']
		#+ (((.002)*df_offense['x_smooth']) -1)
		#- .04*(47-df_offense['x_smooth'])
	camera_y = 65
	camera_x = 47
	k_midcourt = 2  
	k_basket = 0.3    # Higher correction near the basket (further from camera)

# Define scaling factor based on distance from midcourt (closer to basket = higher correction)
	o_scaling_factor_x = 1 - (df_offense['x_smooth'] / 47)
	d_scaling_factor_x = 1 - (df_defense['x_smooth'] / 47)
# Apply a scaled correction factor based on the player's x-position
	df_offense['x_smooth'] = df_offense['x_smooth'] - (k_midcourt + o_scaling_factor_x * (k_basket - k_midcourt)) * np.arctan2(camera_y - df_offense['y_smooth'], df_offense['x_smooth'] - camera_x)
	df_offense['y_smooth'] = df_offense['y_smooth'] - 2 + (df_offense['y_smooth']-25)/20

	df_defense['x_smooth'] = df_defense['x_smooth'] - (k_midcourt + d_scaling_factor_x * (k_basket - k_midcourt)) * np.arctan2(camera_y - df_defense['y_smooth'], df_defense['x_smooth'] - camera_x)
	df_defense['y_smooth'] = df_defense['y_smooth'] - 2 + (df_defense['y_smooth']-25)/20

	df_shooter['x_smooth'] = df_shooter['x_smooth'] 
	df_shooter['y_smooth'] = df_shooter['y_smooth'] 

# Drop the NaN values that result from the rolling operation
	df_offense.dropna(inplace=True)
	df_defense.dropna(inplace=True)
	df_shooter.dropna(inplace=True)

	fig, ax = plt.subplots(figsize=(15, 8))
	create_ncaa_full_court(ax,
						three_line='mens',
						paint_alpha=0.4,
						inner_arc=True,margin=5)
	marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}
	offense, = ax.plot([], [], ms=10, markerfacecolor='#b94b75', **marker_kwargs)  # red/maroon
	defense, = ax.plot([], [], ms=10, markerfacecolor='#7f63b8', **marker_kwargs)  # purple
	shooter, = ax.plot([], [], ms=12, markerfacecolor='yellow', **marker_kwargs)  # yellow for shooter

	if shot_loc == True:
		if example == True:
			df_loc = pd.read_csv('example_shot.csv')
			
		else:
			df = read_locations()
			df_loc = df[(df['game_id'] == game_id) & (df['play_id'] == play_id)]
		

		# Create a dictionary for shot positions
		shot = df_loc[df_loc['annotation_code'] == 's']
		teammates = df_loc[df_loc['annotation_code'].isin(['t1', 't2', 't3', 't4'])]

		# Find the frame where all players are closest to their shooting positions
		best_frame, min_distance = find_best_frame(df_offense, df_shooter, teammates,shot)
		print(f"Best frame where all players are closest to their positions: {best_frame}, Total Distance: {min_distance}")


		if show_closest_frame:
	# Find the positions for the closest frame
	#best_frame
				show_frame(best_frame,df_offense,df_shooter,ax,teammates,shot)
				return

		


	positions_offense = df_offense.groupby('frame')[['x_smooth', 'y_smooth']].apply(lambda x: (x['x_smooth'].values, x['y_smooth'].values)).to_dict()
	positions_defense = df_defense.groupby('frame')[['x_smooth', 'y_smooth']].apply(lambda x: (x['x_smooth'].values, x['y_smooth'].values)).to_dict()
	positions_shooter = df_shooter.groupby('frame')[['x_smooth', 'y_smooth']].apply(lambda x: (x['x_smooth'].values, x['y_smooth'].values)).to_dict()



	def animate(i):
		frame = df_offense.iloc[i, df_offense.columns.get_loc('frame')]
	
	# Set positions from pre-calculated dictionaries
		offense.set_data(*positions_offense.get(frame, ([], [])))
		defense.set_data(*positions_defense.get(frame, ([], [])))
		shooter.set_data(*positions_shooter.get(frame, ([], [])))
		
		for _, row in teammates.iterrows():
				ax.plot(row['court_x'], row['court_y'], 
				marker='x', color='blue', markersize=8, linestyle='None', label=f"Teammate {row['annotation_code']}")

		if not shot.empty:
				ax.plot(shot['court_x'].values[0], shot['court_y'].values[0], 
				marker='x', color='red', markersize=12, linestyle='None', label='Shot Location')

		return offense, defense, shooter

	anim = animation.FuncAnimation(fig, animate, frames=len(df_defense['frame'].unique()), interval=70, blit=True)
	plt.show()


if __name__ == '__main__':
		animate_play(19783001319551,6,example = True, smooth = True,show_closest_frame=True)




#hello

''' sample play data for reference: locations
game_id,play_id,annotation_code,court_x,court_y
19783001319551,6,d1,5.710742501112131,34.12272746746357
19783001319551,6,d2,5.287257698866037,14.042897407825176
19783001319551,6,d3,9.318426664058979,6.612670421600342
19783001319551,6,d4,13.188731725399311,17.685114420377293
19783001319551,6,d5,21.786111244788536,25.203976264366737
19783001319551,6,s,10.767018519915068,1.7216248007921071
19783001319551,6,t1,9.606983093115,45.07533953740047
19783001319551,6,t2,6.590887794127831,14.623409051161547
19783001319551,6,t3,22.388302399561955,12.894430527320274
19783001319551,6,t4,35.14796286362868,23.594487630403957
'''

''' sample play data for reference: tracking
game_id,play_id,type,frame,x,y,tracklet_id
'''
#72324
#72316
#72322
