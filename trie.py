from constants import *


class TrieNode:
    """
    Data structure to represent each Node in Trie
    """
    def __init__(self):
        self.arrChildren = [None] * CNT_ALPHABETS
        self.iCount = 0
        self.bIsWord = False

class Trie:
    """
    Class that encapsulates tree helper functions 
    with Trie root node
    """
    def __init__(self):
        self.objRoot = TrieNode()

    def insertWord(self, strWord):
        """
        Inserts word into Trie
        Args:
            strWord (str): word to be inserted into Trie
        """
        # check if the word given is a valid word
        if strWord is not None and strWord != EMPTY_STRING:
            try:
                # convert the word to lowercase
                strWord = strWord.lower()

                # fetch the root node of the Trie
                objNode = self.objRoot

                # traverse trie to increment word count
                for char in strWord:
                    iIndex = ord(char) - ASCII_A
                    
                    if objNode.arrChildren[iIndex] is None:
                        objNode.arrChildren[iIndex] = TrieNode()
                    
                    objNode = objNode.arrChildren[iIndex]
                
                objNode.bIsWord = True
                objNode.iCount += 1
                    
                return 0
            except:
                raise BaseException('Word: ' + strWord)
        return -1

    def getWordCount(self, strWord):
        """
        Fetches the word count from trie if it exists. 
        Else return -1
        Args:
            strWord (str): word to be inserted into Trie
        """
        if strWord is not None and strWord != EMPTY_STRING:
            strWord = strWord.lower()
            objNode = self.objRoot

            for char in strWord:
                iIndex = ord(char) - ASCII_A

                if objNode.arrChildren[iIndex] is None:
                    return -1
                
                objNode = objNode.arrChildren[iIndex]
            
            if objNode.bIsWord:
                return objNode.iCount
            return -1                
        return -1
    