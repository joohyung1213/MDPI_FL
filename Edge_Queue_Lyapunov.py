import numpy as np
import random
import matplotlib.pyplot as plt
import csv
#from scipy import stats


C_max = 50
T_max = 1000
Q_max = 2000

# Comm
f = 2.5 * (10 ** 9)
Pt = 20  # dBm
exponentN = 3

# Battery

# Data
datasize = 3  # MegaByte
dataPerClient = 2000
datachunk = 10

static_selection_client_number = int(C_max / 12)

V = 10 ** 10

# Client number 정해졌을 때 데이터 수
CN_data_amount = np.arange(C_max+1)
CN_data_amount = CN_data_amount * datachunk

CN_accuracy = np.arange(Q_max+1, dtype='f')
l_rate = 100
d_rate = -0.2
for i in range(1, Q_max+1):
    CN_accuracy[i] = 100 - l_rate * (pow(CN_accuracy[i], d_rate))

T_timeunit = np.arange(T_max)
Q_unit = np.arange(Q_max+1)


# Comm
# C_comm = np.random.rand(C_max)
C_distance = np.zeros(C_max)
C_receive = np.zeros(C_max)
C_comm = np.zeros(C_max)
C_comm = np.full(C_max, 1, dtype=int)


# C_power = np.random.normal(0, 0.1, C_max)
C_power = np.zeros(C_max)
for i in range(C_max):
    C_power[i] = (int)(random.random() * T_max * 0.9)
#min_power = np.abs(np.min(C_power))
C_power_random = C_power.copy()
C_power_static = C_power.copy()


# Data
C_data_proposed = np.zeros(C_max)
C_data_proposed = np.full(C_max, dataPerClient, dtype=int)
C_data_random = C_data_proposed.copy()


C_priority = np.zeros(C_max)
C_fairness_random = np.zeros(C_max)
C_fairness_proposed = np.zeros(C_max)

T_client_selection = np.arange(C_max+1)

T_queue_proposed = np.zeros(T_max)
T_queue_random = np.zeros(T_max)
T_compare_queue1 = np.zeros(T_max)
T_compare_queue2 = np.zeros(T_max)
T_client_choice_proposed = np.zeros(T_max)
T_client_choice_random = np.zeros(T_max)
T_accuracy_proposed = np.zeros(T_max)
T_accuracy_random = np.zeros(T_max)
T_departure = np.zeros(T_max)
T_alive_client_proposed = np.zeros(T_max)
T_alive_client_random = np.zeros(T_max)
Total_Data_Proposed = 0
Total_Data_Random = 0


for t in range(T_max):
    print(t)

    departure = random.random()
    if departure < 0.95:
        departure = 10 * datachunk * random.random()
    else:
        departure = 0


    max_choice_client = 0
    max_choice_data = 0

    for i in range(C_max):
        C_distance[i] = random.random() * 100
        C_receive[i] = 20 - (20 * np.log10(f) + 10 * exponentN * np.log10(C_distance[i]) - 28)
        #C_receive[i] = 10 ** (C_receive[i] / 10)
        C_comm[i] = C_receive[i] * random.random()
    C_comm_min = min(C_comm)
    if C_comm_min < 0:
        C_comm = C_comm - C_comm_min


# Proposed Number Decision
    if t == 0:
        max_choice_client_Proposed = C_max
        max_choice_data_Proposed = max_choice_client_Proposed * datachunk

        max_choice_client_Random = C_max
        max_choice_data_Random = max_choice_client_Random * datachunk

        T_queue_proposed[t] = 0
        T_queue_random[t] = 0
        T_compare_queue1[t] = 0
        T_compare_queue2[t] = 0
    else:
        val_proposed = -9999999999
        val_temp_proposed = val_proposed - 1
        for i in range(C_max + 1):
            if 0 < (T_queue_proposed[t - 1] + CN_data_amount[i]) and (T_queue_proposed[t - 1] + CN_data_amount[i]) < Q_max + 1:
                val_temp_proposed = V * CN_accuracy[(int)(T_queue_proposed[t - 1] + CN_data_amount[i])] - T_queue_proposed[t - 1] * CN_data_amount[i]
                if val_temp_proposed > val_proposed:
                    val_proposed = val_temp_proposed
                    max_choice_client_Proposed = i

        val_random = -9999999999
        val_temp_random = val_random - 1
        for i in range(C_max + 1):
            if 0 < (T_queue_random[t - 1] + CN_data_amount[i]) and (T_queue_random[t - 1] + CN_data_amount[i]) < Q_max + 1:
                val_temp_random = V * CN_accuracy[(int)(T_queue_random[t - 1] + CN_data_amount[i])] - T_queue_random[t - 1] * CN_data_amount[i]
                if val_temp_random > val_random:
                    val_random = val_temp_random
                    max_choice_client_Random = i

        # num_alive_client = sum(x[0] > 0 and x[1] > 0 for x in enumerate(zip(C_power, C_data)))

# Proposed Selection
        num_alive_client_Proposed = 0
        for x in zip(C_power, C_data_proposed):
            if x[0] > 0 and x[1] > 0:
                num_alive_client_Proposed += 1
        max_choice_client_Proposed = min(max_choice_client_Proposed, num_alive_client_Proposed)
        alive_client_Proposed = np.argwhere((C_power > 0) & (C_data_proposed > 0))
        alive_client_Proposed = alive_client_Proposed.reshape(num_alive_client_Proposed)

        for i in range(C_max):
            if (C_power[i] <= 0) or (C_data_proposed[i] <= 0):
                C_priority[i] = 0
            else:
                C_priority[i] = C_data_proposed[i] * C_comm[i] / C_power[i]
                # C_priority[i] = C_comm_norm[i] / C_power_norm[i]
                # C_priority[i] = 1 / C_power_norm[i]

        client_choice_Proposed = C_priority.argsort()[::-1][:max_choice_client_Proposed]

        for i in client_choice_Proposed:
            C_data_proposed[i] -= datachunk
            C_fairness_proposed[i] += 1
            Total_Data_Proposed += datachunk

# Random Select
        num_alive_client_Random = 0
        for x in zip(C_power_random, C_data_random):
            if x[0] > 0 and x[1] > 0:
                num_alive_client_Random += 1
        max_choice_client_Random = min(max_choice_client_Random, num_alive_client_Random)
        alive_client_Random = np.argwhere((C_power_random > 0) & (C_data_random > 0))
        alive_client_Random = alive_client_Random.reshape(num_alive_client_Random)

        client_choice_Random = random.sample(list(alive_client_Random), max_choice_client_Random)

        for i in client_choice_Random:
            C_data_random[i] -= datachunk
            C_fairness_random[i] += 1
            Total_Data_Random += datachunk

# Static Select
        num_alive_client_Static = 0
        for x in C_power_static:
            if x > 0:
                num_alive_client_Static += 1
        static_selection_client_number = min(static_selection_client_number, num_alive_client_Static)


        for i in range(C_max):
            C_power[i] -= 1
            C_power_random[i] -= 1
            C_power_static[i] -= 1

        max_choice_data_Proposed = max_choice_client_Proposed * datachunk
        max_choice_data_Random = max_choice_client_Random * datachunk

        T_queue_proposed[t] = max(T_queue_proposed[t-1] + max_choice_data_Proposed - departure, 0)
        T_queue_random[t] = max(T_queue_random[t-1] + max_choice_data_Random - departure, 0)
        T_compare_queue1[t] = T_compare_queue1[t-1] + (C_max * datachunk) - departure
        T_compare_queue2[t] = max(T_compare_queue2[t-1] + static_selection_client_number * datachunk - departure, 0)
        T_client_choice_proposed[t] = max_choice_client_Proposed
        T_client_choice_random[t] = max_choice_client_Random
        T_accuracy_proposed[t] = CN_accuracy[(int)(T_queue_proposed[t])]
        T_accuracy_random[t] = CN_accuracy[(int)(T_queue_random[t])]
        T_departure[t] = departure
        T_alive_client_proposed[t] = num_alive_client_Proposed
        T_alive_client_random[t] = num_alive_client_Random


f1 = open('Proposed_power.csv', 'w')
wr = csv.writer(f1)
wr.writerow(T_queue_proposed)

f2 = open('Random_power.csv', 'w')
wr = csv.writer(f2)
wr.writerow(T_queue_random)

f3 = open('Full_power.csv', 'w')
wr = csv.writer(f3)
wr.writerow(T_compare_queue1)

f4 = open('Static_power.csv', 'w')
wr = csv.writer(f4)
wr.writerow(T_compare_queue2)

f5 = open('Fairness_proposed.csv', 'w')
wr = csv.writer(f5)
wr.writerow(C_fairness_proposed)

f6 = open('Fairness_random.csv', 'w')
wr = csv.writer(f6)
wr.writerow(C_fairness_random)

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()

interval = 3

print("Total Data : ", dataPerClient * C_max)
print("Total_Data_Proposed : ", Total_Data_Proposed)
print("Total_Data_Random : ", Total_Data_Random)


plt.figure(1)
plt.axhline(y=Q_max, color='k', linestyle='--', linewidth=3.0)
plt.ylim(0, Q_max * 5)
plt.xlim(0, T_max)


line1, = plt.plot(T_timeunit, T_queue_proposed[:], label='Proposed')
line2, = plt.plot(T_timeunit, T_queue_random[:], label='Random')
line3, = plt.plot(T_timeunit, T_compare_queue1[:], label='full')
line4, = plt.plot(T_timeunit, T_compare_queue2[:], label='static')

plt.setp(line1, color='r', linewidth=4.0)
plt.setp(line2, color='g', linewidth=2.5)
plt.setp(line3, color='k', linewidth=4.0)
plt.setp(line4, color='b', linewidth=4.0)
plt.legend(handles=(line1, line2, line3, line4), labels=('Proposed', 'Random', 'Full', 'Static number selection'), prop={'size':30})
plt.xlabel('Time Slot')
plt.ylabel('Queue Backlog')
plt.grid(True)
# plt.setp(line4, color='k', linewidth=1.0)


plt.figure(7)
# histogram = plt.hist([C_fairness_proposed, C_fairness_random])
# n, bins, _ = plt.hist([C_fairness_proposed, C_fairness_random], bins=np.arange(-1, max(max(C_fairness_proposed), max(C_fairness_random)))+1)
n, bins, _ = plt.hist([C_fairness_proposed, C_fairness_random], bins=np.arange(-1, 200)+1, label=['Proposed', 'Random'])

device_count_proposed = np.zeros(201)
for x in range(len(C_fairness_proposed)):
    device_count_proposed[int(C_fairness_proposed[x])] += 1

device_count_random = np.zeros(201)
for x in range(len(C_fairness_random)):
    device_count_random[int(C_fairness_random[x])] += 1

yaxis_lim = max(max(device_count_random), max(device_count_proposed))
print(yaxis_lim)

f7 = open('Occurance.csv', 'w')
wr = csv.writer(f7)
wr.writerow(n)
f7.close()
# plt.title('Client Number per Communication Number')
plt.legend(prop={'size': 25})
plt.xlabel('Number of Communication', fontsize=30, fontname='Arial')
plt.ylabel('Number of Client', fontsize=30, fontname='Arial')
plt.xticks(np.arange(0, 201, 20))
plt.yticks(np.arange(0, yaxis_lim + 1, 1))
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.margins(x=0.01)

# =========================================================================
plt.figure(8)
# histogram = plt.hist([C_fairness_proposed, C_fairness_random])
# n, bins, _ = plt.hist([C_fairness_proposed, C_fairness_random], bins=np.arange(-1, max(max(C_fairness_proposed), max(C_fairness_random)))+1)
C_fairness_proposed_5 = C_fairness_proposed.copy()
for x in range(len(C_fairness_proposed)):
    C_fairness_proposed_5[x] = int(np.round(C_fairness_proposed[x]/5)*5)

C_fairness_random_5 = C_fairness_random.copy()
for x in range(len(C_fairness_random)):
    C_fairness_random_5[x] = int(np.round(C_fairness_random[x]/5)*5)

# n, bins, _ = plt.hist([C_fairness_proposed_5, C_fairness_random_5], bins=np.arange(-1, 40)+1, label=['Proposed', 'Random'])
n, bins, _ = plt.hist([C_fairness_proposed_5, C_fairness_random_5], bins=np.arange(-1, 200, 5)+1, label=['Proposed', 'Random'])

device_count_proposed_5 = np.zeros(201)
for x in range(len(C_fairness_proposed_5)):
    device_count_proposed_5[int(C_fairness_proposed_5[x])] += 1

device_count_random_5 = np.zeros(201)
for x in range(len(C_fairness_random_5)):
    device_count_random_5[int(C_fairness_random_5[x])] += 1

yaxis_lim = max(max(device_count_proposed_5), max(device_count_random_5))
print(yaxis_lim)

f8 = open('Occurance_5.csv', 'w')
wr = csv.writer(f8)
wr.writerow(n)
f8.close()
# plt.title('Client Number per Communication Number')
plt.legend(prop={'size': 25})
plt.xlabel('Number of Communication', fontsize=30, fontname='Arial')
plt.ylabel('Number of Client', fontsize=30, fontname='Arial')
plt.xticks(np.arange(0, 201, 20))
plt.yticks(np.arange(0, yaxis_lim + 1, 1))
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.margins(x=0.01)


# plt.grid(True)
plt.show()
