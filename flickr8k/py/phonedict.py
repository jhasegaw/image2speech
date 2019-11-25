import sys,os


phonefile=sys.argv[1]
dictfile=sys.argv[2]

# Load the phones
phones=set()
with open(phonefile) as f:
    for line in phonefile:
        if len(line.rstrip()) > 0:
            phones.add(line.rstrip())

# Write the dictionary
with open(dictfile,'w') as f:
    f.write('MNCL\n')
    for p in sorted(phones):
        # Say that something is stressed if it's a vowel, but not AX or IX
        if len(p)>1 and p[0] in 'AEIOU' and p[1]!='X':
            f.write('("%s" nil (((%s) 1)))\n' % (p,p))
        else:
            f.write('("%s" nil (((%s) 1)))\n' % (p,p))
