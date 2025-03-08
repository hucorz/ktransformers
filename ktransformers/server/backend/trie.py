from ktransformers.server.config.log import logger


class Trie:
    def __init__(self):
        self.nex = [{}]
        self.is_end = [False]
        self.category = [None]

    def build(self, categories):
        for category in categories:
            self.insert(category)
        # logger.info(f"Trie build done: {self.nex}")

    def insert(self, category):
        words = str(category)
        # words = category.split()
        p = 0
        for word in words:
            if word not in self.nex[p]:
                self.nex[p][word] = len(self.nex)
                self.nex.append({})
                self.is_end.append(False)
                self.category.append(None)
            p = self.nex[p][word]
        self.is_end[p] = True
        self.category[p] = category

    def search_unique_category(self, prefix):
        # logger.info(f"Search unique category:\n{prefix}\n")
        prefix = prefix.strip()
        if prefix[0] == '"':
            prefix = prefix[1:]
        # prefix_words = prefix.split()
        prefix_words = prefix
        p = 0
        for word in prefix_words:
            if word not in self.nex[p]:
                return None
            p = self.nex[p][word]

        def count_children(idx):
            return len(self.nex[idx])

        if count_children(p) == 0 and self.is_end[p]:
            return self.category[p]

        while count_children(p) == 1:
            next_word = next(iter(self.nex[p]))
            p = self.nex[p][next_word]
            if self.is_end[p]:
                return self.category[p]
        return None


def build_trie_from_format(format: list[str]):
    trie = Trie()
    trie.build(format)
    return trie


if __name__ == "__main__":
    categories = [
        "hello world",
        "hello there friend",
        "good morning world",
        "good morning sunshine",
        "welcome to python",
        "welcome to programming",
        "learn python basics",
        "learn programming fundamentals",
    ]

    trie = Trie()
    trie.build(categories)

    prefix = "learn programming"
    result = trie.search_unique_category(prefix)
    print("唯一匹配的类别:" if result else "仍然有歧义", result)

    prefix = "welcome"
    result = trie.search_unique_category(prefix)
    print("唯一匹配的类别:" if result else "仍然有歧义", result)

    prefix = "hello there"
    result = trie.search_unique_category(prefix)
    print("唯一匹配的类别:" if result else "仍然有歧义", result)
