def reverseVowels(s):
    """
    :type s: str
    :rtype: str
    """
    vowels = ['a', 'e', 'i', 'o','u','A', 'E', 'I', 'O','U']
    s = list(s)
    left = 0 
    right = len(s)-1
    print(right)
    temp = None
    while left < right:
        
        if((s[left] in vowels) and (s[right] in vowels)):
            print(s[left])
            temp = s[left]
            s[left] = s[right]
            s[right] = temp

            left+=1
            right-=1
        elif((s[left] in vowels) and (s[right] not in vowels)):
            right -= 1

        elif((s[left] not in vowels) and (s[right] in vowels)):
            left += 1

    return "".join(s)


print(reverseVowels("IceCreAm"))