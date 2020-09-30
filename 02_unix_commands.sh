curl -O https://nlp100.github.io/data/popular-names.txt
# q 10
wc -l popular-names.txt
# q 11
expand -t1 popular-names.txt | head -10
# q 12
!cut -f 1 popular-names.txt  > col1.txt
!cut -f 2 popular-names.txt  > col2.txt
# q 13
!paste col* > col.txt
# q 14
split -n3 data
# q 15
tail -n10 xaa
# q 16
cut -f 1 data | sort | uniq
# q 17
sort -k 3 -r data

# q 18

# q 19

