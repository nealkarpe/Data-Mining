import csv

def rowContainsItemSet(row, itemset):
	"""(Bool) Check if a row contains a particular itemset"""
	for item in itemset:
		if row[item] != '1':
			return False
	return True

def frequency(str_item_set,data):
	"""(Int) Returns the number of rows the itemset is present in"""
	items = list(map(int, str_item_set.split("#")))
	count = 0
	for row in data:
		if rowContainsItemSet(row, items):
			count += 1
	return count

def join(string1, string2):
	"""Input: two itemsets (in string format) of same length
	   Output: returns union itemset if it is length+1, else returns false"""
	itemset1 = set(list(map(int, string1.split("#"))))
	len1 = len(itemset1)
	itemset2 = set(list(map(int, string2.split("#"))))
	union = itemset1.union(itemset2)
	if len(union) != len1 + 1:
		return False
	num_list = list(union)
	num_list.sort()
	str_list = [str(item) for item in num_list]
	return "#".join(str_list)

def frequentItemSets(items,data,min_sup):
	"""return value is a set of itemsets (in string format)"""
	result = {}
	k = 1
	candidateItemSets = set([str(item) for item in items])
	while len(candidateItemSets) != 0 and k<=len(items):
		verifiedItemSets = set()
		for str_item_set in candidateItemSets:
			item_set_frequency = frequency(str_item_set,data)
			if item_set_frequency >= min_sup:
				verifiedItemSets.add(str_item_set)
				result[str_item_set] = item_set_frequency
		candidateItemSets = set()
		verifiedSetList = list(verifiedItemSets)
		n = len(verifiedItemSets)
		if n == 0:
			break
		print("count of "+str(k)+"-item frequent itemsets:", n)
		for i in range(n-1):
			for j in range(i+1,n):
				joined_str_item_set = join(verifiedSetList[i], verifiedSetList[j])
				if joined_str_item_set:
					candidateItemSets.add(joined_str_item_set)
		k += 1
	return result

def parseResult(result, attributes_row):
	list_of_strings = list(result)
	list_of_lists = list(map(lambda x: list(map(int,x.split("#"))), list_of_strings))
	list_of_lists.sort()
	list_of_lists.sort(key=len)
	print("---------------Frequent itemsets along with their support count---------------")
	for numericItemList in list_of_lists:
		strItemList = "#".join(list(map(str,numericItemList)))
		support_count = result[strItemList]
		namedItemList = list(map(lambda x: attributes_row[x], numericItemList))
		print(", ".join(namedItemList), "- " + str(support_count))

items = [1,2,3,4,5,6,7,8,9,10,11,12,14,15,16]
data = []
with open('zoo.csv', 'r') as f:
	reader = list(csv.reader(f))
	attributes_row = list(reader[0])
	for i in range(1,len(reader)):
		data.append(list(reader[i]))

min_sup = 50 # change min support count value here
result = frequentItemSets(items, data, min_sup) # set of all itemsets (in string format)
parseResult(result, attributes_row)
