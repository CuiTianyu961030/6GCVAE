# -*- coding: utf-8 -*-


# 读取数据
def read_data(filename):
    f = open(filename, 'r', encoding='utf-8')

    # 去除 gasser_ipv6hitlist 第一行的 '# Responsive IPv6 addresses, published on 2019-08-03'
    addresses = f.readlines()[1:]
    f.close()
    for i in range(len(addresses)):
        addresses[i] = addresses[i][:-1]

    return addresses


# 对地址进行填充，使其为标准128位地址的4bit-segment格式
def flatten(addresses):
    print('flattening addresses in source data...')
    flatten_addresses = []

    for address in addresses:
        nybbles = address.split(':')

        # 恢复前导0
        for i in range(len(nybbles)):
            if len(nybbles[i]) < 4:
                for _ in range(4 - len(nybbles[i])):
                    nybbles[i] = '0' + nybbles[i]
        # 双冒号填充
        if len(nybbles) < 8:
            for i in range(8):
                if nybbles[i] == '0000':
                    for _ in range(8 - len(nybbles)):
                        nybbles.insert(i, '0000')
        sign = ':'
        flatten_addresses.append(sign.join(nybbles))

    print('finished example:')
    print('source: ', addresses[0], 'flattened: ', flatten_addresses[0])

    return flatten_addresses


# 生成训练数据集并存储flatten地址
def data_save(addresses, address_class):

    # dataset_path = 'data/processed_data/' + address_class + '_measurement.txt'
    # addresses_path = 'data/processed_data/' + address_class + '_measurement_flatten_addresses.txt'

    dataset_path = 'data/processed_data/data.txt'
    addresses_path = 'data/processed_data/flatten_addresses.txt'

    dataset = []
    flatten_addresses = []
    for address in addresses:
        address = address + '\n'
        raw_data = address.split(':')
        sign = ''
        raw_data = sign.join(raw_data)
        dataset.append(raw_data)
        flatten_addresses.append(address)

    f = open(dataset_path, 'w', encoding='utf-8')
    f.writelines(dataset)
    f.close()

    f = open(addresses_path, 'w', encoding='utf-8')
    f.writelines(flatten_addresses)
    f.close()

    print('processed data saved path: ', dataset_path)
    print('flattened addresses saved path: ', addresses_path)


# 地址手工分类
def address_classification(addresses):
    fixed_iid_addresses = []
    low_64bit_subnet_addresses = []
    slaac_eui64_addresses = []
    slaac_privacy_addresses = []
    father_temp = ""
    for address in addresses:
        # print(address)
        if '::' in address:
            last_string = address.split('::')[-1]
            if ':' not in last_string:
                fixed_iid_addresses.append(address)
                father_temp = address
                continue
        else:
            if address.split(':')[-2][0:2] == 'fe' and address.split(':')[-3][2:4] == 'ff':
                slaac_eui64_addresses.append(address)
                father_temp = address
                continue
            elif len(address.split(':')[-1]) >= 3 and len(address.split(':')[-2]) >= 3 and\
                    len(address.split(':')[-3]) >= 3 and len(address.split(':')[-4]) >= 3 and\
                    len(father_temp.split(':')) >= 4 and\
                    father_temp.split(':')[-2] != address.split(':')[-2] and\
                    father_temp.split(':')[-3] != address.split(':')[-3] and\
                    father_temp.split(':')[-4] != address.split(':')[-4]:
                slaac_privacy_addresses.append(address)
                father_temp = address
                continue
        low_64bit_subnet_addresses.append(address)
        father_temp = address

    return fixed_iid_addresses, low_64bit_subnet_addresses, slaac_eui64_addresses, slaac_privacy_addresses


def cluster_process(cluster_path):
    prefix = []
    total_prefix = []
    f = open(cluster_path, "r")
    for line in f:

        if line[0] == "=":
            prefix = []
            continue
        elif line[0] == "\n":
            total_prefix.append(prefix)
            continue
        if line[0] == "S":
            break
        prefix.append(line.split(" ")[0])

    f.close()
    # print(total_prefix[1])

    data_path = "data/processed_data/gasser_data_1107.txt"

    cluster_data = []
    for cluster in total_prefix:
        cluster_i = []
        for prefix in cluster:
            f = open(data_path, "r")
            for line in f:
                if prefix == line[:8]:
                    cluster_i.append(line)
            f.close()
        cluster_data.append(cluster_i)

    # print(cluster_data[0])

    count = 0
    for cluster in cluster_data:
        f = open("data/processed_data/gasser_cluster_" + str(count) + "_1107.txt", "w")
        f.writelines(cluster)
        f.close()
        count += 1


if __name__ == '__main__':
    gasser_ipv6hitlist = 'data/public_datasets/responsive-addresses.txt'
    #
    addresses = read_data(gasser_ipv6hitlist)
    fixed_iid_addresses, low_64bit_subnet_addresses, slaac_eui64_addresses, slaac_privacy_addresses = \
        address_classification(addresses)
    #
    # flatten_fixed_iid_addresses = flatten(fixed_iid_addresses)
    # flatten_low_64bit_subnet_addresses = flatten(low_64bit_subnet_addresses)
    # flatten_slaac_eui64_addresses = flatten(slaac_eui64_addresses)
    # flatten_slaac_privacy_addresses = flatten(slaac_privacy_addresses)
    flatten_total_data = flatten(addresses)
    #
    # data_save(flatten_fixed_iid_addresses, address_class="fixed_iid_addresses")
    # data_save(flatten_low_64bit_subnet_addresses, address_class="low_64bit_subnet_addresses")
    # data_save(flatten_slaac_eui64_addresses, address_class="slaac_eui64_addresses")
    # data_save(flatten_slaac_privacy_addresses, address_class="slaac_privacy_addresses")
    data_save(flatten_total_data, address_class="")

    # cluster process
    # cluster_path = "/Users/cuitianyu/Tools/entropy-clustering/clusters_1107.txt"
    # cluster_process(cluster_path)
