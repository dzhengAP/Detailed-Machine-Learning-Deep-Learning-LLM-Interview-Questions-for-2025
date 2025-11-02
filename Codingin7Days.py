# Day 1
# arry fundamentals and two pointers
# 1 tow sum 输入:nums = [2,7,11,15], target = 9 输出：[0,1]  # 2+7=9
def towSum(self, nums,target):
    hashmap={}
    for i, num in enumerate(nums):
        if target - num in hashmap:
            return [hashmap[target-num], i]
        hashmap[num]=i
# 2 Remove Duplicates from Sorted Array — LC #26
# 输入：nums = [1,1,2]
# 输出：2  # nums 前2个是 [1,2]
def removeDup(self, nums):
    if not nums:
        return 0
    slow+=1
    for fast in range(1, len(nums)):
        if nums[fast] != nums[fast-1]:
            nums[slow]=nums[fast]
            slow+=1
    return slow
    
✅ LC 1 — Two Sum
Knowledge：Hash table + complement lookup
Interview Test Point：How to reduce O(n²) to O(n)
✅ Example
Input: nums=[2,7,11,15], target=9
Output: [0,1]
✅ Key Idea (Hint)
For each element x, check if target - x seen before
✅ Code (Line-by-line Comments)
class Solution:
    def twoSum(self, nums, target):
        hashmap = {}  # value -> index
        for i, x in enumerate(nums):
            diff = target - x
            if diff in hashmap:  # found complement
                return [hashmap[diff], i]
            hashmap[x] = i  # store index of x
✅ Similar Problems (Key Hints)
LC#	Title	Hint
167	Two Sum II	sorted → two pointers
653	Two Sum IV BST	DFS + set
454	4Sum II	map pair sums
560	Subarray Sum Equals K	prefix sum + hash
===================================================
✅ LC 26 — Remove Duplicates from Sorted Array
Knowledge：Sorted + Fast/Slow Pointers
Test Point：In-place dedup & pointer control
✅ Example
Input: [1,1,2]
Output: 2  # nums=[1,2,...]
✅ Hint
Sorted → duplicates are consecutive
✅ Code
class Solution:
    def removeDuplicates(self, nums):
        if not nums: return 0
        slow = 1
        for fast in range(1, len(nums)):
            if nums[fast] != nums[fast - 1]:
                nums[slow] = nums[fast]
                slow += 1
        return slow
✅ Similar Problems
LC#	Description	Hint
27	Remove Element	same pointer idea
80 ✅	Allow ≤2 duplicates	compare slow-2
283	Move Zeroes	swap non-zero forward
===================================================
✅ LC 88 — Merge Sorted Array
Knowledge：Tail merge to avoid shifting
Test Point：Pointer direction matters
✅ Example
nums1=[1,2,3,0,0,0], nums2=[2,5,6]
Output=[1,2,2,3,5,6]
✅ Hint
Merge from back → O(m+n)
✅ Code
class Solution:
    def merge(self, nums1, m, nums2, n):
        i, j, k = m - 1, n - 1, m + n - 1
        while j >= 0:
            if i >= 0 and nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1
✅ Similar Problems
LC#	Description	Hint
21	Merge 2 sorted lists	linked list version
23	Merge K lists	min-heap
986	Interval intersection	sweep both lists
===================================================
✅ LC 283 — Move Zeroes
Knowledge：Stable in-place partition
Test Point：Swap order & O(n)
✅ Example
Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
✅ Hint
Swap non-zeros forward, preserve order
✅ Code
class Solution:
    def moveZeroes(self, nums):
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1
✅ Similar Problems
LC#	Description	Hint
905	Sort by parity	even-forward
75	Sort colors	3-way partition

✅ LC 27 — Remove Element
Knowledge：Filter out unwanted values
Test Point：Overwrite vs remove
✅ Example
nums=[3,2,2,3], val=3 → output:2, nums=[2,2]
✅ Hint
fast scans; slow writes survivors
✅ Code
class Solution:
    def removeElement(self, nums, val):
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
        return slow
===================================================
✅ LC 125 — Valid Palindrome
Knowledge：Character validation + two pointers
Test Point：Filtering efficiently
✅ Example
"A man, a plan, a canal: Panama" → true
✅ Hint
Move inward, skip non-alnum
✅ Code
class Solution:
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        while l < r:
            while l < r and not s[l].isalnum(): l += 1
            while l < r and not s[r].isalnum(): r -= 1
            if s[l].lower() != s[r].lower(): return False
            l += 1; r -= 1
        return True
✅ Similar Problems
LC#	Description	Hint
680	Palindrome II	allow 1 removal
844	Backspace compare	filter simulation
===================================================
✅ LC 167 — Two Sum II (Sorted)
Knowledge：Sorted + Two Pointers (low/high)
Test Point：Pointer direction based on compare
✅ Example
[2,7,11,15], target=9 → [1,2]
✅ Code
class Solution:
    def twoSum(self, nums, target):
        i, j = 0, len(nums) - 1
        while i < j:
            s = nums[i] + nums[j]
            if s == target: return [i+1, j+1]
            elif s < target: i += 1
            else: j -= 1
✅ Similar Problems
LC#	Description	Hint
15	3Sum	sort + two pointers

✅ LC 15 — 3Sum
Knowledge：Sort + dedup + two-pointer search
Test Point：Avoid duplicates properly
✅ Example
Input: [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
✅ Code
class Solution:
    def threeSum(self, nums):
        nums.sort()
        res = []
        n = len(nums)
        for i in range(n):
            if i > 0 and nums[i] == nums[i-1]: continue
            l, r = i + 1, n - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s == 0:
                    res.append([nums[i], nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while l < r and nums[l] == nums[l-1]: l += 1
                    while l < r and nums[r] == nums[r+1]: r -= 1
                elif s < 0: l += 1
                else: r -= 1
        return res
#259	3Sum Smaller	bounding
#Given an array of n integers nums and an integer target, find the number of index triplets i, j, k with 0 <= i < j < k < n that satisfy the condition nums[i] + nums[j] + nums[k] < target.

#Example 1:
#Input: nums = [-2,0,1,3], target = 2
#Output: 2
#Explanation: Because there are two triplets which sums are less than 2:
#[-2,0,1]
#[-2,0,3]
#✅ Code
class Solution:
    def threeSumSmaller(self, nums):
        nums.sort()
        ans=0
        for i in range(len(nums)):
            l = i+1
            r = len(nums)-1
            while l<r:
                if nums[i]+nums[l]+nums[r] < target:
                    ans+=r-l
                    l+=1
                else:
                    r-=1
        return ans
✅ Similar Problems
#LC#	Description	Hint
#16	3Sum closest	minimize
#Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target.
#Return the sum of the three integers.
#You may assume that each input would have exactly one solution.

'''

Example 1:

Input: nums = [-1,2,1,-4], target = 1
Output: 2
Explanation: The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
'''
class Solution:
    def threeSumSmaller(self, nums):
        nums.sort()
        delta, ans = float('inf'), 0
        for i in range(len(nums)):
            l, r = i+1, len(nums)-1
            while l < r:
                if abs(target-(nums[i]+ nums[j]+ nums[k]))< delta:
                    ans = target-(nums[i]+ nums[j]+ nums[k])
                    delta = abs(ans)
                    r-=1
        return ans
        

===================================================
✅ Day 1 Techniques You Mastered
Technique	Used in
Hash Map for complement	LC1, LC653
Fast/Slow pointers	LC26, LC27, LC283, LC905
Two pointers (sorted)	LC167, LC977, LC15
Tail merge	LC88
String filter pointers	LC125, LC844
Sorting + dedup logic	LC15, LC80
===================================================

✅ DAY 2 — Sliding Window & Prefix Sum
🔹 LC 209 — Minimum Size Subarray Sum
找到最短的连续子数组，使其和 ≥ target
关键思路
滑动窗口（可变长度）
右指针扩张满足条件后，左指针缩小
Example
输入: target=7, nums=[2,3,1,2,4,3]
输出: 2  # [4,3]
Solution
class Solution:
    def minSubArrayLen(self, target: int, nums: list[int]) -> int:
        l = 0
        s = 0
        res = float('inf')

        for r in range(len(nums)):
            s += nums[r]  # 扩大窗口加总

            # 当当前窗口和 >= target，则尝试缩小
            while s >= target:
                res = min(res, r - l + 1)
                s -= nums[l]  # 缩小窗口
                l += 1

        return 0 if res == float('inf') else res
✅ Pattern: Expand until valid → Shrink to optimize
🔹 LC 3 — Longest Substring Without Repeating Characters
最长无重复字符子串
Key Thinking
Keep the window unique
哈希记录字符最近出现位置
Example
输入: "abcabcbb"
输出: 3  # "abc"
Solution
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        last = {}
        l = ans = 0

        for r, ch in enumerate(s):
            # 如果 ch 在当前窗口出现过
            if ch in last and last[ch] >= l:
                l = last[ch] + 1

            last[ch] = r
            ans = max(ans, r - l + 1)

        return ans
✅ Pattern: unique window, use dict to track last index
🔹 LC 76 — Minimum Window Substring
找到覆盖字符串 t 所有字符的最小窗口
Key Thinking
maintain counts: need vs window
valid window = window[ch] >= need[ch]
Example
输入: s="ADOBECODEBANC", t="ABC"
输出: "BANC"
Solution
from collections import Counter

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = Counter(t)
        window = {}
        have = 0
        required = len(need)
        
        l = 0
        res, res_len = [-1, -1], float('inf')

        for r, ch in enumerate(s):
            window[ch] = window.get(ch, 0) + 1

            if ch in need and window[ch] == need[ch]:
                have += 1

            # 当所有字符都满足时，尝试缩小窗口
            while have == required:
                if (r - l + 1) < res_len:
                    res = [l, r]
                    res_len = r - l + 1

                window[s[l]] -= 1
                if s[l] in need and window[s[l]] < need[s[l]]:
                    have -= 1
                l += 1

        l, r = res
        return s[l:r+1] if res_len != float('inf') else ""
✅ One of the most important sliding window templates! 🔥
🔹 LC 560 — Subarray Sum Equals K
连续子数组和 = k 的数量
Key Thinking
Prefix sum where:
ps[j] - ps[i] = k → ps[i] = ps[j] - k
Example
输入: nums=[1,1,1], k=2
输出: 2  # [1,1] twice
Solution
from collections import defaultdict

class Solution:
    def subarraySum(self, nums, k):
        count = defaultdict(int)
        count[0] = 1  # 前缀和为0出现一次

        ps = 0
        res = 0

        for x in nums:
            ps += x
            res += count[ps - k]  # 找到配对前缀和
            count[ps] += 1

        return res
✅ Template: 前缀和 + 哈希统计
🔹 LC 523 — Continuous Subarray Sum
求一个子数组和：sum % k == 0 且长度 ≥ 2
Key Thinking
状态相同模值之间的区间长度≥2
Example
输入: [23,2,4,6,7], k=6
输出: True
Solution
class Solution:
    def checkSubarraySum(self, nums, k):
        mp = {0: -1}  # 初始模状态位置
        ps = 0

        for i, x in enumerate(nums):
            ps = (ps + x) % k

            if ps in mp:
                if i - mp[ps] >= 2:
                    return True
            else:
                mp[ps] = i  # 只记录最早位置
                
        return False
✅ Prefix sum modulo trick
🔹 LC 974 — Subarrays Divisible by K
统计满足 sum % k == 0 的子数组个数
Example
输入: [4,5,0,-2,-3,1], k=5
输出: 7
Solution
class Solution:
    def subarraysDivByK(self, nums, k):
        cnt = [0] * k
        cnt[0] = 1

        ps = ans = 0

        for x in nums:
            ps = (ps + x) % k
            ans += cnt[ps]  # 已有相同模值的次数
            cnt[ps] += 1

        return ans
✅ Same idea as LC523 → counting equal remainder pairs
🔹 LC 438 — Find All Anagrams in a String
找所有 t 异位词起始位置
Key Thinking
Fixed length sliding window
Compare char counts
Example
输入: s="cbaebabacd", t="abc"
输出: [0,6]
Solution
from collections import Counter

class Solution:
    def findAnagrams(self, s, t):
        if len(t) > len(s): 
            return []

        need = Counter(t)
        window = Counter()
        l = 0
        res = []

        for r, ch in enumerate(s):
            window[ch] += 1

            if r - l + 1 > len(t):
                window[s[l]] -= 1
                if window[s[l]] == 0:
                    del window[s[l]]
                l += 1

            if window == need:
                res.append(l)

        return res
✅ Fixed window size + count compare

✅ DAY 2 Reinforcement Practice Pack
Topic: Sliding Window + Prefix Sums
🔹 LC 424 — Longest Repeating Character Replacement
允许最多 k 次替换，使子串全部相同 → 求最长长度
✅ How to think
Sliding window: maintain most frequent char count
Condition: window_len - max_freq > k → invalid
✅ Example
输入: s = "ABAB", k = 2
输出: 4  # "ABAB" 全替成 A 或 B
✅ Solution
from collections import Counter

class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        count = Counter()
        l = 0
        maxf = 0  # 当前窗口内最多字符数
        res = 0

        for r, ch in enumerate(s):
            count[ch] += 1
            maxf = max(maxf, count[ch])

            # 若替换次数超出 k，收缩窗口
            while (r - l + 1) - maxf > k:
                count[s[l]] -= 1
                l += 1

            res = max(res, r - l + 1)

        return res
✅ Template: shrink when condition violated
🔥 This is asked in FAANG frequently
🔹 LC 567 — Permutation in String
s1 是否为 s2 某子串的排列（异位词匹配）
✅ Key Thinking
Fixed length sliding window
Char count match
✅ Example
输入: s1 = "ab", s2 = "eidbaooo"
输出: true  # "ba" is permutation of "ab"
✅ Solution
from collections import Counter

class Solution:
    def checkInclusion(self, s1, s2):
        need = Counter(s1)
        window = Counter()
        k = len(s1)
        l = 0

        for r, ch in enumerate(s2):
            window[ch] += 1

            # 只保持固定窗口大小
            if r - l + 1 > k:
                window[s2[l]] -= 1
                if window[s2[l]] == 0:
                    del window[s2[l]]
                l += 1

            if window == need:
                return True

        return False
✅ Same pattern as LC438
🎯 Good practice for anagram sliding windows

🔹 LC 1004 — Max Consecutive Ones III
最多替换 k 个 0 → 让连续 1 最长
✅ Why similar
Same as LC424 but boolean condition
Count zeros in window
✅ Example
输入: nums=[1,1,1,0,0,0,1,1,1,1,0], k=2
输出: 6
✅ Solution
class Solution:
    def longestOnes(self, nums, k):
        l = 0
        zero_count = 0
        ans = 0
        
        for r, x in enumerate(nums):
            if x == 0:
                zero_count += 1
            
            # 若窗口内 0 太多 → 收缩
            while zero_count > k:
                if nums[l] == 0:
                    zero_count -= 1
                l += 1
            
            ans = max(ans, r - l + 1)

        return ans
✅ Binary array version of LC424 pattern
🔹 LC 30 — Substring with Concatenation of All Words
所有单词等长 → 子串包含全部单词（无重复顺序）
✅ Key Insight
Split fixed size chunks inside sliding window
Compare word counts
✅ Example
输入: s="barfoothefoobarman", words=["foo","bar"]
输出: [0,9]
✅ Solution
from collections import Counter

class Solution:
    def findSubstring(self, s, words):
        if not s or not words:
            return []

        word_len = len(words[0])
        total_len = word_len * len(words)
        need = Counter(words)
        res = []

        for i in range(word_len):
            l = i
            window = Counter()
            count = 0

            for r in range(i, len(s), word_len):
                word = s[r:r+word_len]

                if word in need:
                    window[word] += 1
                    count += 1

                    # 若某单词出现过多，收缩窗口
                    while window[word] > need[word]:
                        left_word = s[l:l+word_len]
                        window[left_word] -= 1
                        l += word_len
                        count -= 1

                    if count == len(words):
                        res.append(l)

                else:
                    window.clear()
                    count = 0
                    l = r + word_len

        return res
✅ High value problem — multiple patterns combined

🔹 LC 930 — Binary Subarrays With Sum
子数组和 = goal 的数量（binary array）
✅ Why it fits Day2
Prefix sum + frequency
Exact-sum counting (like LC560)
✅ Example
输入: nums = [1,0,1,0,1], goal = 2
输出: 4
✅ Solution

from collections import defaultdict
class Solution:
    def numSubarraysWithSum(self, nums, goal):
        count = defaultdict(int)
        count[0] = 1

        ps = ans = 0
        for x in nums:
            ps += x
            ans += count[ps - goal]
            count[ps] += 1
        
        return ans
✅ Special case of LC560 for 0/1 arrays

✅ DAY 3 — Binary Search + Greedy
These two topics are MUST-MASTER for interviews 🔥
Binary Search → solve search / decision
Greedy → solve optimal interval / scheduling
🔹 LC 35 — Search Insert Position
返回 target 插入位置（保持升序）
✅ How to think
Binary search 左闭右开模板
→ 找第一个 ≥ target 的位置（lower bound）
✅ Example
输入: nums=[1,3,5,6], target=5
输出: 2
✅ Code + Comments
class Solution:
    def searchInsert(self, nums, target):
        l, r = 0, len(nums)  # 搜索区间 [l, r)

        while l < r:
            mid = (l + r) // 2
            if nums[mid] < target:
                l = mid + 1  # target 在右边
            else:
                r = mid  # mid 可能是答案

        return l  # 最终 l 就是插入位置
✅ Template: Lower Bound 二分法
🔹 LC 34 — Find First and Last Position of Element
返回 target 连续出现区间
✅ Pattern
两次二分：
✅ 找左边界
✅ 找右边界
✅ Example
输入: [5,7,7,8,8,10], target=8
输出: [3,4]
✅ Code
class Solution:
    def searchRange(self, nums, target):
        def bound(isLeft):
            l, r = 0, len(nums)
            while l < r:
                mid = (l + r) // 2
                if nums[mid] > target or (isLeft and nums[mid] == target):
                    r = mid
                else:
                    l = mid + 1
            return l
        
        left = bound(True)
        if left == len(nums) or nums[left] != target:
            return [-1, -1]
        right = bound(False) - 1
        return [left, right]
✅ Don’t forget edge case return check
🔹 LC 153 — Find Minimum in Rotated Sorted Array
旋转过的升序数组 → 找最小值
✅ How to think
Binary search judging sorted half
✅ Example
输入: [3,4,5,1,2]
输出: 1
✅ Code
class Solution:
    def findMin(self, nums):
        l, r = 0, len(nums) - 1

        while l < r:
            mid = (l + r) // 2
            if nums[mid] > nums[r]:
                l = mid + 1  # 最小值在右边
            else:
                r = mid  # mid 可能是最小
        return nums[l]
✅ Key: Compare with right pointer
🔹 LC 162 — Find Peak Element
找任意一个峰值（nums[i] > nums[i+1]）
✅ Insight
Always move towards higher side
✅ Example
输入: [1,2,3,1]
输出: 2  # 3的位置
✅ Code
class Solution:
    def findPeakElement(self, nums):
        l, r = 0, len(nums)-1
        while l < r:
            mid = (l + r) // 2
            if nums[mid] < nums[mid+1]:
                l = mid + 1  # 上坡，往右走
            else:
                r = mid  # mid 可能是峰
        return l
✅ 山坡形状 → 往上走
🔹 LC 300 — Longest Increasing Subsequence
LIS 长度
✅ Trick ✅✅✅
🛑 DP O(n²)太慢
✅ Binary search + Greedy → O(n log n)
✅ Example
输入: [10,9,2,5,3,7,101,18]
输出: 4  # [2,3,7,18]
✅ Code
import bisect

class Solution:
    def lengthOfLIS(self, nums):
        tails = []  # tails[i] = 长度为 i+1 的递增序列的最小末尾
        for x in nums:
            i = bisect.bisect_left(tails, x)
            if i == len(tails):
                tails.append(x)
            else:
                tails[i] = x
        return len(tails)
✅ tails 是一个 状态压缩后的 DP
🔹 LC 435 — Non-overlapping Intervals
删除最少区间 → 使无重叠
✅ Greedy rule
优先选择右端点最小的区间
✅ Example
输入: [[1,2],[2,3],[3,4],[1,3]]
输出: 1
✅ Code
class Solution:
    def eraseOverlapIntervals(self, intervals):
        intervals.sort(key=lambda x: x[1])
        cnt = 0
        end = float('-inf')

        for s, e in intervals:
            if s >= end:
                end = e
            else:
                cnt += 1  # 删除当前区间
        return cnt
✅ Classic interval scheduling
🔹 LC 452 — Minimum Number of Arrows to Burst Balloons
贪心区间覆盖
✅ Insight
排序 by END
每射一箭 → 尽量覆盖更多
✅ Example
输入: [[10,16],[2,8],[1,6],[7,12]]
输出: 2
✅ Code
class Solution:
    def findMinArrowShots(self, points):
        points.sort(key=lambda x: x[1])
        arrows = 1
        end = points[0][1]

        for s, e in points[1:]:
            if s > end:  # 无法覆盖，需新箭
                arrows += 1
                end = e
        return arrows
✅ Same logic as LC435 ✔
🔹 LC 410 — Split Array Largest Sum
分割数组为 k 段 → 最小化最大段和
✅ Pattern: Binary Search on Answer
Check(mid) 可行性 → 缩小答案区间
✅ Example
输入: [7,2,5,10,8], k=2
输出: 18
组: [7,2,5] and [10,8]
✅ Code
class Solution:
    def splitArray(self, nums, k):
        def can(mid):
            s = 0
            cnt = 1
            for x in nums:
                if s + x > mid:
                    cnt += 1
                    s = 0
                s += x
            return cnt <= k
        
        l, r = max(nums), sum(nums)
        while l < r:
            mid = (l + r) // 2
            if can(mid):
                r = mid
            else:
                l = mid + 1
        return l
✅ Answer is monotonic → Binary Search applies
✅ Day 3 Skills Summary
Skill	Where Applied	Notes
二分查找 (lower bound)	LC35 LC34	最小满足条件的索引
二分找峰值	LC162	往上坡方向移动
二分答案	LC410	can(mid) 单调可行性
区间贪心	LC435 LC452	按右端点排序最优
LIS 优化	LC300	贪心 + binary search


ChatGPT said:
Absolutely! ✅ Here is Day 4 — continuing the same premium “Full-Learning Package” format:
✅ LeetCode Number
✅ Problem Summary
✅ Key Reasoning (“How to think!”)
✅ Example Input/Output
✅ Full Python Code with line-by-line comments
✅ Skills takeaway

✅ DAY 4 — Trees & Graphs (DFS / BFS)
This is one of the most common interview themes 🔥
If you master recursion + graph traversal here → many hard problems become easy.
🌳 LC 104 — Maximum Depth of Binary Tree
求最大深度（层数）
✅ How to think
Use DFS（后序遍历）
→ Depth = 1 + max(left depth, right depth)
✅ Example
输入: [3,9,20,null,null,15,7]
输出: 3
✅ Code + Explanation
class Solution:
    def maxDepth(self, root):
        if not root:
            return 0  # 空节点深度为 0

        # 分别求左右子树深度
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)

        # 当前节点深度 = 左右最大值 + 1
        return 1 + max(left, right)
✅ Classic recursion structure
🎯 Key trick: post-order traversal
🌳 LC 226 — Invert Binary Tree
左右子节点互换（镜像树）
✅ Example
输入: [4,2,7,1,3,6,9]
输出: [4,7,2,9,6,3,1]
✅ Code + Comments
class Solution:
    def invertTree(self, root):
        if not root:
            return None

        # 交换左右子树
        root.left, root.right = root.right, root.left

        # 递归处理子树
        self.invertTree(root.left)
        self.invertTree(root.right)

        return root
✅ Simple but often asked — tests your recursion fundamentals
🏝️ LC 200 — Number of Islands
统计“岛屿”数量
(grid[i][j] == '1' → 陆地)
✅ Key Thinking
通过 DFS 将全岛 flood fill 为 '0'
每次 DFS 表示遇到新岛屿
✅ Example
11000
11000
00100
00011
输出: 3
✅ Code
class Solution:
    def numIslands(self, grid):
        if not grid:
            return 0

        rows, cols = len(grid), len(grid[0])
        visited = set()

        def dfs(r, c):
            # 越界 or 已访问 or 水 → 停止
            if (r < 0 or r >= rows or
                c < 0 or c >= cols or
                grid[r][c] == '0' or
                (r, c) in visited):
                return

            visited.add((r, c))

            # 四个方向继续探索
            dfs(r+1, c)
            dfs(r-1, c)
            dfs(r, c+1)
            dfs(r, c-1)

        islands = 0
        for i in range(rows):
            for j in range(cols):
                # 找到新的岛屿
                if grid[i][j] == '1' and (i, j) not in visited:
                    islands += 1
                    dfs(i, j)

        return islands
✅ Classic 4-direction DFS flood-fill
🌊 LC 695 — Max Area of Island
找最大岛屿的大小
✅ Similar to LC200 but returns largest connected cell count
✅ Code
class Solution:
    def maxAreaOfIsland(self, grid):
        rows, cols = len(grid), len(grid[0])
        visited = set()

        def dfs(r, c):
            if (r < 0 or r >= rows or
                c < 0 or c >= cols or
                grid[r][c] == 0 or
                (r,c) in visited):
                return 0

            visited.add((r, c))

            return (1 +
                dfs(r+1,c) +
                dfs(r-1,c) +
                dfs(r,c+1) +
                dfs(r,c-1)
            )

        area = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    area = max(area, dfs(i, j))
        return area
✅ Only adds counting to LC200 logic
🌲 LC 102 — Binary Tree Level Order Traversal (BFS)
层序遍历 → 每层一行
✅ Example
输入: [3,9,20,null,null,15,7]
输出: [[3],[9,20],[15,7]]
✅ BFS Code
from collections import deque

class Solution:
    def levelOrder(self, root):
        if not root:
            return []

        q = deque([root])
        res = []

        while q:
            size = len(q)
            level = []

            for _ in range(size):
                node = q.popleft()
                level.append(node.val)

                # 加入下一层节点
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)

            res.append(level)

        return res
✅ BFS = queue
🎯 structured layering
🔁 LC 133 — Clone Graph
复制 图结构（neighbors deep copy）
✅ Key Thinking
Graph may have cycles 🌀
→ must track visited mapping old→new
✅ Code
class Solution:
    def cloneGraph(self, node):
        if not node:
            return None

        visited = {}

        def dfs(n):
            if n in visited:
                return visited[n]
            
            # 克隆节点
            copy = Node(n.val)
            visited[n] = copy

            # 克隆邻居
            for nei in n.neighbors:
                copy.neighbors.append(dfs(nei))
            
            return copy

        return dfs(node)
✅ Critical: Must prevent infinite loop!
✅ Day 4 Key Skills Summary
Skill	Where Learned	Notes
DFS recursion	LC104, LC200, LC695	Most frequent interview style
BFS queue	LC102	Good for level-order / shortest path
Visited set	LC200, LC695, LC133	Prevent revisiting/cycles
Tree basics	LC104, LC226	Master recursion patterns
🌟 If you can do Day 4 smoothly → graphs & trees become fun instead of scary.

ChatGPT said:
Absolutely! ✅ Here is Day 4 — continuing the same premium “Full-Learning Package” format:
✅ LeetCode Number
✅ Problem Summary
✅ Key Reasoning (“How to think!”)
✅ Example Input/Output
✅ Full Python Code with line-by-line comments
✅ Skills takeaway

✅ DAY 4 — Trees & Graphs (DFS / BFS)
This is one of the most common interview themes 🔥
If you master recursion + graph traversal here → many hard problems become easy.
🌳 LC 104 — Maximum Depth of Binary Tree
求最大深度（层数）
✅ How to think
Use DFS（后序遍历）
→ Depth = 1 + max(left depth, right depth)
✅ Example
输入: [3,9,20,null,null,15,7]
输出: 3
✅ Code + Explanation
class Solution:
    def maxDepth(self, root):
        if not root:
            return 0  # 空节点深度为 0

        # 分别求左右子树深度
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)

        # 当前节点深度 = 左右最大值 + 1
        return 1 + max(left, right)
✅ Classic recursion structure
🎯 Key trick: post-order traversal
🌳 LC 226 — Invert Binary Tree
左右子节点互换（镜像树）
✅ Example
输入: [4,2,7,1,3,6,9]
输出: [4,7,2,9,6,3,1]
✅ Code + Comments
class Solution:
    def invertTree(self, root):
        if not root:
            return None

        # 交换左右子树
        root.left, root.right = root.right, root.left

        # 递归处理子树
        self.invertTree(root.left)
        self.invertTree(root.right)

        return root
✅ Simple but often asked — tests your recursion fundamentals
🏝️ LC 200 — Number of Islands
统计“岛屿”数量
(grid[i][j] == '1' → 陆地)
✅ Key Thinking
通过 DFS 将全岛 flood fill 为 '0'
每次 DFS 表示遇到新岛屿
✅ Example
11000
11000
00100
00011
输出: 3
✅ Code
class Solution:
    def numIslands(self, grid):
        if not grid:
            return 0

        rows, cols = len(grid), len(grid[0])
        visited = set()

        def dfs(r, c):
            # 越界 or 已访问 or 水 → 停止
            if (r < 0 or r >= rows or
                c < 0 or c >= cols or
                grid[r][c] == '0' or
                (r, c) in visited):
                return

            visited.add((r, c))

            # 四个方向继续探索
            dfs(r+1, c)
            dfs(r-1, c)
            dfs(r, c+1)
            dfs(r, c-1)

        islands = 0
        for i in range(rows):
            for j in range(cols):
                # 找到新的岛屿
                if grid[i][j] == '1' and (i, j) not in visited:
                    islands += 1
                    dfs(i, j)

        return islands
✅ Classic 4-direction DFS flood-fill
🌊 LC 695 — Max Area of Island
找最大岛屿的大小
✅ Similar to LC200 but returns largest connected cell count
✅ Code
class Solution:
    def maxAreaOfIsland(self, grid):
        rows, cols = len(grid), len(grid[0])
        visited = set()

        def dfs(r, c):
            if (r < 0 or r >= rows or
                c < 0 or c >= cols or
                grid[r][c] == 0 or
                (r,c) in visited):
                return 0

            visited.add((r, c))

            return (1 +
                dfs(r+1,c) +
                dfs(r-1,c) +
                dfs(r,c+1) +
                dfs(r,c-1)
            )

        area = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    area = max(area, dfs(i, j))
        return area
✅ Only adds counting to LC200 logic
🌲 LC 102 — Binary Tree Level Order Traversal (BFS)
层序遍历 → 每层一行
✅ Example
输入: [3,9,20,null,null,15,7]
输出: [[3],[9,20],[15,7]]
✅ BFS Code
from collections import deque

class Solution:
    def levelOrder(self, root):
        if not root:
            return []

        q = deque([root])
        res = []

        while q:
            size = len(q)
            level = []

            for _ in range(size):
                node = q.popleft()
                level.append(node.val)

                # 加入下一层节点
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)

            res.append(level)

        return res
✅ BFS = queue
🎯 structured layering
🔁 LC 133 — Clone Graph
复制 图结构（neighbors deep copy）
✅ Key Thinking
Graph may have cycles 🌀
→ must track visited mapping old→new
✅ Code
class Solution:
    def cloneGraph(self, node):
        if not node:
            return None

        visited = {}

        def dfs(n):
            if n in visited:
                return visited[n]
            
            # 克隆节点
            copy = Node(n.val)
            visited[n] = copy

            # 克隆邻居
            for nei in n.neighbors:
                copy.neighbors.append(dfs(nei))
            
            return copy

        return dfs(node)
✅ Critical: Must prevent infinite loop!
✅ Day 4 Key Skills Summary
Skill	Where Learned	Notes
DFS recursion	LC104, LC200, LC695	Most frequent interview style
BFS queue	LC102	Good for level-order / shortest path
Visited set	LC200, LC695, LC133	Prevent revisiting/cycles
Tree basics	LC104, LC226	Master recursion patterns
🌟 If you can do Day 4 smoothly → graphs & trees become fun instead of scary.

You said:
please add comment for each line
ChatGPT said:
Got it ✅ You want Day 7 full line-by-line commented versions (every line explained clearly).
Here they are — all Day 7 solutions upgraded ✅

✅ Day 7 — Full Line-By-Line Commented Code
✅ LC 155 — Min Stack
class MinStack:
    def __init__(self):
        # 主栈：存所有值
        self.st = []
        # 辅助栈：存当前最小值轨迹
        self.mn = []

    def push(self, x):
        # 压入主栈
        self.st.append(x)
        # 如果辅助栈是空 或 x 比当前最小值小，则更新辅助栈
        if not self.mn or x <= self.mn[-1]:
            self.mn.append(x)

    def pop(self):
        # 弹出主栈顶部值
        x = self.st.pop()
        # 如果这个值是当前最小值，则辅助栈同步弹出
        if x == self.mn[-1]:
            self.mn.pop()

    def top(self):
        # 返回主栈顶部元素
        return self.st[-1]

    def getMin(self):
        # 返回辅助栈顶部 → 当前最小值
        return self.mn[-1]
✅ O(1) getMin()
✅ 必会
✅ LC 146 — LRU Cache
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        # 缓存容量
        self.cap = capacity
        # OrderedDict 维护插入顺序（LRU）
        self.od = OrderedDict()

    def get(self, key: int) -> int:
        # key 不在缓存中 → 返回 -1
        if key not in self.od:
            return -1
        # 将访问的 key 移到队尾（最近使用）
        self.od.move_to_end(key)
        return self.od[key]

    def put(self, key: int, value: int) -> None:
        # 如果 key 已存在 → 更新并挪到队尾
        if key in self.od:
            self.od.move_to_end(key)
        # 写入或更新值
        self.od[key] = value
        # 如果超过容量 → 移除最久未使用的键（队头）
        if len(self.od) > self.cap:
            self.od.popitem(last=False)
✅ OrderedDict = Hash + 双链表
🎯 高频考点
✅ LC 215 — Kth Largest Element in an Array
import heapq  # Python 默认是最小堆

class Solution:
    def findKthLargest(self, nums, k):
        h = []  # 最小堆
        for x in nums:
            # 推入一个元素
            heapq.heappush(h, x)
            # 如果堆大小超过 k → 弹掉最小值
            if len(h) > k:
                heapq.heappop(h)
        # 堆顶即第 k 大值
        return h[0]
✅ O(n log k)，比排序更快
✅ 面试常考
✅ LC 23 — Merge K Sorted Lists
import heapq

class Solution:
    def mergeKLists(self, lists):
        pq = []  # 最小堆 (value, list_index, node)

        # 初始化：将每个链表的第一节点加入堆
        for i, node in enumerate(lists):
            if node:
                heapq.heappush(pq, (node.val, i, node))

        dummy = ListNode(0)  # 虚拟头节点
        cur = dummy  # 当前处理指针

        while pq:
            # 弹出最小值节点
            _, i, node = heapq.heappop(pq)
            # 接入结果链表
            cur.next = node
            cur = cur.next
            # 如果该链表还有下一个节点 → 推入堆
            if node.next:
                heapq.heappush(pq, (node.next.val, i, node.next))

        # 返回合并后链表
        return dummy.next
✅ Heap 合并 K 路排序流
🔥 高频 Hard
You said:
add problem example solution with comment
ChatGPT said:
Got it ✅
You want each Day 7 problem with:
✔ Problem statement (short & clear)
✔ Example input/output
✔ Fully commented Python solution (every line explained)

I will complete all remaining problems from Day 7 in this message:

✅ LC 239 Sliding Window Maximum
✅ LC 84 Largest Rectangle in Histogram
✅ LC 42 Trapping Rain Water

(You already have LC155 / LC146 / LC215 / LC23 fully commented above, so we continue.)

✅ Day 7 (Remaining Problems Fully Expanded)
🔹LC 239 — Sliding Window Maximum
每个长度为 k 的滑窗内的最大值
✅ Example
输入:
nums = [1,3,-1,-3,5,3,6,7], k = 3
输出: [3,3,5,5,6,7]
✅ Key Idea
Monotonic Queue (Deque)
→ 保持队列内 值从大到小
→ 队头永远是最大值
✅ Code with Full Comments
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums, k):
        dq = deque()  # 存储下标，保证对应值递减
        res = []      # 结果数组
        
        for i, x in enumerate(nums):
            # 1️⃣ 弹出队尾小于当前值的 → 没用
            while dq and nums[dq[-1]] <= x:
                dq.pop()
            
            # 把当前元素下标加入
            dq.append(i)
            
            # 2️⃣ 移除窗口左边界以外的下标
            if dq[0] <= i - k:
                dq.popleft()
            
            # 3️⃣ 当形成窗口后，记录窗口最大值
            if i >= k - 1:
                res.append(nums[dq[0]])
        
        return res
✅ O(n)，每个元素最多入队出队一次
🔹LC 84 — Largest Rectangle in Histogram
给直方图求最大矩形面积
✅ Example
输入: [2,1,5,6,2,3]
输出: 10   # 5x2 矩形
✅ Key Idea
Monotonic increasing stack
→ 当当前高度 < 栈顶高度 → 计算栈顶能形成的最大矩形
✅ Code with Full Comments
class Solution:
    def largestRectangleArea(self, heights):
        stack = []        # 单调递增栈，存索引
        maxA = 0          # 最大面积记录
        
        heights.append(0) # 加 sentinel 强制结算
        
        for i, h in enumerate(heights):
            # 出现下降 → 计算面积
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]  # 当前能形成矩形的高
                left = stack[-1] if stack else -1  # 左边界
                width = i - left - 1  # 宽度取两边界差
                maxA = max(maxA, height * width)
            
            stack.append(i)  # 入栈索引
        
        heights.pop() # 恢复原数组
        return maxA
✅ O(n) — 最多入栈出栈一次
🔹LC 42 — Trapping Rain Water
计算能接多少雨水
✅ Example
输入: [0,1,0,2,1,0,1,3,2,1,2,1]
输出: 6
✅ Key Idea
Two pointers
left_max = 左边最高挡板
right_max = 右边最高挡板
较低挡板决定能装多少水
✅ Code with Full Comments
class Solution:
    def trap(self, height):
        l, r = 0, len(height)-1  # 两个指针
        left_max = right_max = 0
        ans = 0
        
        while l < r:
            if height[l] < height[r]:
                # 更新左最大高度
                if height[l] >= left_max:
                    left_max = height[l]
                else:
                    # 可以接到 left_max - 当前高度 的水
                    ans += left_max - height[l]
                l += 1
            else:
                # 更新右最大高度
                if height[r] >= right_max:
                    right_max = height[r]
                else:
                    ans += right_max - height[r]
                r -= 1
        
        return ans
✅ 比 DP 前缀数组节省空间
✅ 面试常考、逻辑性强
✅ Day 7 Completion Summary ✅
Problem	Pattern
LC155	Two stacks track minimum
LC146	Hash + Doubly Linked List
LC215	Heap with size K
LC23	Heap merge sorted streams
⭐ LC239	Monotonic queue
⭐ LC84	Monotonic stack
⭐ LC42	Two pointer water trap
🌟 LC239 / LC84 / LC42 → Top-tier interview must-masters
