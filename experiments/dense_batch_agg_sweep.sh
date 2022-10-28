#!/bin/bash
# Parameter sweep of batch size and aggregation dimension
cd ~/graphsage-sparse

# BTER
n=1
d='bter'
p='gpu'
a='dense'
bs=(128 512 1024)
e=5
ads=(256 512 1024)
de=2
m=100
nss=5
s=15
do=.5

echo 'BTER'
for b in "${bs[@]}"; do
    echo "Batch Size:  ${b}"
    for ad in "${ads[@]}"; do
        echo "Agg Dim:  ${ad}"
        jobname="${d}_a_${a}_b_${b}_ad_${ad}_m_${m}_s_${s}"
        python graphsage.py -n ${n} -a ${a} -d ${d} -p ${p} -b ${b} -e ${e} -ad ${ad} -de ${de} -m ${m} -nss ${nss} -s ${s} -do ${do} >"./output/${jobname}.out.txt" 2>"./output/${jobname}.err.txt"
    done
done

#LREDDIT
n=1
d='lreddit'
p='gpu'
a='dense'
bs=(256 512 1024)
e=5
ads=(128 512 1024)
de=2
m=128
nss=5
s=15
do=.5
pt=.01

echo 'LREDDIT'
for b in "${bs[@]}"; do
    echo "Batch Size:  ${b}"
    for ad in "${ads[@]}"; do
        echo "Agg Dim:  ${ad}"
        jobname="${d}_a_${a}_b_${b}_ad_${ad}_m_${m}_s_${s}"
        python graphsage.py -n ${n} -a ${a} -d ${d} -p ${p} -b ${b} -e ${e} -ad ${ad} -de ${de} -m ${m} -nss ${nss} -s ${s} -do ${do} -pt ${pt} >"./output/${jobname}.out.txt" 2>"./output/${jobname}.err.txt"
    done
done

#ARXIV
n=1
d='arxiv'
p='gpu'
a='dense'
bs=(128 512 1024)
e=200
ads=(256 512 1024)
de=2
m=100
s=15
pa=20
do=.5

echo 'ARXIV'
for b in "${bs[@]}"; do
    echo "Batch Size:  ${b}"
    for ad in "${ads[@]}"; do
        echo "Agg Dim:  ${ad}"
        jobname="${d}_a_${a}_b_${b}_ad_${ad}_m_${m}_s_${s}"
        python graphsage.py -n ${n} -a ${a} -d ${d} -p ${p} -b ${b} -e ${e} -ad ${ad} -de ${de} -m ${m} -s ${s} -do ${do} -pa ${pa} >"./output/${jobname}.out.txt" 2>"./output/${jobname}.err.txt"
    done
done

#NREDDIT
n=1
d='nreddit'
p='gpu'
a='dense'
bs=(256 512 1024)
e=200
ads=(128 512 1024)
de=2
m=128
s=15
pa=20
do=.5

echo 'NREDDIT'
for b in "${bs[@]}"; do
    echo "Batch Size:  ${b}"
    for ad in "${ads[@]}"; do
        echo "Agg Dim:  ${ad}"
        jobname="${d}_a_${a}_b_${b}_ad_${ad}_m_${m}_s_${s}"
        python graphsage.py -n ${n} -a ${a} -d ${d} -p ${p} -b ${b} -e ${e} -ad ${ad} -de ${de} -m ${m} -s ${s} -do ${do} -pa ${pa} >"./output/${jobname}.out.txt" 2>"./output/${jobname}.err.txt"
    done
done
