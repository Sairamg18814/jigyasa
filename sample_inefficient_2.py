
def find_duplicates(items):
    """Find duplicate items in a list"""
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                if items[i] not in duplicates:
                    duplicates.append(items[i])
    return duplicates

def merge_lists(list1, list2):
    """Merge two lists inefficiently"""
    merged = []
    for item in list1:
        merged.append(item)
    for item in list2:
        merged.append(item)
    return merged

def count_occurrences(text, word):
    """Count word occurrences inefficiently"""
    count = 0
    words = text.split()
    for i in range(len(words)):
        if words[i] == word:
            count = count + 1
    return count
