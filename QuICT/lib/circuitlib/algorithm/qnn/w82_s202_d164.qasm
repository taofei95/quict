OPENQASM 2.0;
include "qelib1.inc";
qreg q[82];
creg c[82];
x q[3];
x q[5];
x q[6];
x q[8];
x q[9];
x q[13];
x q[14];
x q[17];
x q[20];
x q[21];
x q[22];
x q[24];
x q[27];
x q[28];
x q[29];
x q[32];
x q[35];
x q[36];
x q[37];
x q[38];
x q[42];
x q[50];
x q[52];
x q[53];
x q[56];
x q[57];
x q[58];
x q[59];
x q[60];
x q[61];
x q[63];
x q[66];
x q[72];
x q[76];
x q[77];
x q[78];
x q[80];
x q[0];
h q[0];
rxx(0.19039475917816162) q[0], q[81];
rxx(0.700595498085022) q[1], q[81];
rxx(0.8554083108901978) q[2], q[81];
rxx(0.2976588010787964) q[3], q[81];
rxx(0.6467397212982178) q[4], q[81];
rxx(0.8584639430046082) q[5], q[81];
rxx(0.5332288146018982) q[6], q[81];
rxx(0.8871553540229797) q[7], q[81];
rxx(0.18393462896347046) q[8], q[81];
rxx(0.601640522480011) q[9], q[81];
rxx(0.5436248183250427) q[10], q[81];
rxx(0.5345235466957092) q[11], q[81];
rxx(0.5046130418777466) q[12], q[81];
rxx(0.748612105846405) q[13], q[81];
rxx(0.18606674671173096) q[14], q[81];
rxx(0.6450093984603882) q[15], q[81];
rxx(0.971514105796814) q[16], q[81];
rxx(0.8641502261161804) q[17], q[81];
rxx(0.28213322162628174) q[18], q[81];
rxx(0.5346282124519348) q[19], q[81];
rxx(0.28359049558639526) q[20], q[81];
rxx(0.34873294830322266) q[21], q[81];
rxx(0.689598560333252) q[22], q[81];
rxx(0.6231103539466858) q[23], q[81];
rxx(0.3502921462059021) q[24], q[81];
rxx(0.5599241852760315) q[25], q[81];
rxx(0.890169084072113) q[26], q[81];
rxx(0.22276413440704346) q[27], q[81];
rxx(0.5345085859298706) q[28], q[81];
rxx(0.9731905460357666) q[29], q[81];
rxx(0.28915178775787354) q[30], q[81];
rxx(0.5649577975273132) q[31], q[81];
rxx(0.7610700130462646) q[32], q[81];
rxx(0.06165182590484619) q[33], q[81];
rxx(0.31573039293289185) q[34], q[81];
rxx(0.17729514837265015) q[35], q[81];
rxx(0.40066754817962646) q[36], q[81];
rxx(0.5991211533546448) q[37], q[81];
rxx(0.1557997465133667) q[38], q[81];
rxx(0.37523096799850464) q[39], q[81];
rxx(0.6953525543212891) q[40], q[81];
rxx(0.48989152908325195) q[41], q[81];
rxx(0.23010462522506714) q[42], q[81];
rxx(0.516852617263794) q[43], q[81];
rxx(0.9638814330101013) q[44], q[81];
rxx(0.06693047285079956) q[45], q[81];
rxx(0.7088366150856018) q[46], q[81];
rxx(0.062475740909576416) q[47], q[81];
rxx(0.663935124874115) q[48], q[81];
rxx(0.7802384495735168) q[49], q[81];
rxx(0.4718576669692993) q[50], q[81];
rxx(0.8308942317962646) q[51], q[81];
rxx(0.08866852521896362) q[52], q[81];
rxx(0.8326655626296997) q[53], q[81];
rxx(0.03749215602874756) q[54], q[81];
rxx(0.13603425025939941) q[55], q[81];
rxx(0.425423800945282) q[56], q[81];
rxx(0.7184328436851501) q[57], q[81];
rxx(0.13288182020187378) q[58], q[81];
rxx(0.5003856420516968) q[59], q[81];
rxx(0.8210433721542358) q[60], q[81];
rxx(0.6407134532928467) q[61], q[81];
rxx(0.2509440779685974) q[62], q[81];
rxx(0.19020813703536987) q[63], q[81];
rxx(0.6837314963340759) q[64], q[81];
rxx(0.09177690744400024) q[65], q[81];
rxx(0.026179015636444092) q[66], q[81];
rxx(0.9336423873901367) q[67], q[81];
rxx(0.2222369909286499) q[68], q[81];
rxx(0.5950087308883667) q[69], q[81];
rxx(0.5882932543754578) q[70], q[81];
rxx(0.12038624286651611) q[71], q[81];
rxx(0.6135753989219666) q[72], q[81];
rxx(0.02216649055480957) q[73], q[81];
rxx(0.13343602418899536) q[74], q[81];
rxx(0.19324111938476562) q[75], q[81];
rxx(0.8313159346580505) q[76], q[81];
rxx(0.23289287090301514) q[77], q[81];
rxx(0.4583587646484375) q[78], q[81];
rxx(0.3615487813949585) q[79], q[81];
rxx(0.03526651859283447) q[80], q[81];
rzz(0.2602270841598511) q[0], q[81];
rzz(0.3596445918083191) q[1], q[81];
rzz(0.6314024925231934) q[2], q[81];
rzz(0.39998364448547363) q[3], q[81];
rzz(0.8338974118232727) q[4], q[81];
rzz(0.11247974634170532) q[5], q[81];
rzz(0.07531261444091797) q[6], q[81];
rzz(0.19981396198272705) q[7], q[81];
rzz(0.5119912028312683) q[8], q[81];
rzz(0.3534078001976013) q[9], q[81];
rzz(0.2061671018600464) q[10], q[81];
rzz(0.035319507122039795) q[11], q[81];
rzz(0.7653735280036926) q[12], q[81];
rzz(0.11649972200393677) q[13], q[81];
rzz(0.5417247414588928) q[14], q[81];
rzz(0.5980826616287231) q[15], q[81];
rzz(0.5639025568962097) q[16], q[81];
rzz(0.9879182577133179) q[17], q[81];
rzz(0.20087331533432007) q[18], q[81];
rzz(0.9232692122459412) q[19], q[81];
rzz(0.6356242895126343) q[20], q[81];
rzz(0.7524664402008057) q[21], q[81];
rzz(0.6716462969779968) q[22], q[81];
rzz(0.45028525590896606) q[23], q[81];
rzz(0.8457390069961548) q[24], q[81];
rzz(0.34445661306381226) q[25], q[81];
rzz(0.4595491886138916) q[26], q[81];
rzz(0.28018951416015625) q[27], q[81];
rzz(0.13264364004135132) q[28], q[81];
rzz(0.06276947259902954) q[29], q[81];
rzz(0.18627184629440308) q[30], q[81];
rzz(0.47842007875442505) q[31], q[81];
rzz(0.3875260353088379) q[32], q[81];
rzz(0.9940130114555359) q[33], q[81];
rzz(0.6449993848800659) q[34], q[81];
rzz(0.3874565362930298) q[35], q[81];
rzz(0.7796100974082947) q[36], q[81];
rzz(0.8386454582214355) q[37], q[81];
rzz(0.9945029020309448) q[38], q[81];
rzz(0.8087637424468994) q[39], q[81];
rzz(0.3678211569786072) q[40], q[81];
rzz(0.5394530296325684) q[41], q[81];
rzz(0.35768330097198486) q[42], q[81];
rzz(0.2649794816970825) q[43], q[81];
rzz(0.48986828327178955) q[44], q[81];
rzz(0.4579848051071167) q[45], q[81];
rzz(0.7510345578193665) q[46], q[81];
rzz(0.5820185542106628) q[47], q[81];
rzz(0.05485564470291138) q[48], q[81];
rzz(0.6761617660522461) q[49], q[81];
rzz(0.6580488681793213) q[50], q[81];
rzz(0.601006269454956) q[51], q[81];
rzz(0.8523240685462952) q[52], q[81];
rzz(0.3375919461250305) q[53], q[81];
rzz(0.01834428310394287) q[54], q[81];
rzz(0.16397130489349365) q[55], q[81];
rzz(0.4070165753364563) q[56], q[81];
rzz(0.6833502054214478) q[57], q[81];
rzz(0.3649963140487671) q[58], q[81];
rzz(0.7892158031463623) q[59], q[81];
rzz(0.5364619493484497) q[60], q[81];
rzz(0.7193725109100342) q[61], q[81];
rzz(0.26247638463974) q[62], q[81];
rzz(0.8261595368385315) q[63], q[81];
rzz(0.9964120984077454) q[64], q[81];
rzz(0.03508031368255615) q[65], q[81];
rzz(0.14745157957077026) q[66], q[81];
rzz(0.43445640802383423) q[67], q[81];
rzz(0.2637540102005005) q[68], q[81];
rzz(0.7093290090560913) q[69], q[81];
rzz(0.07932084798812866) q[70], q[81];
rzz(0.5280596017837524) q[71], q[81];
rzz(0.17316311597824097) q[72], q[81];
rzz(0.2455701231956482) q[73], q[81];
rzz(0.3236096501350403) q[74], q[81];
rzz(0.743953287601471) q[75], q[81];
rzz(0.6939867734909058) q[76], q[81];
rzz(0.04277139902114868) q[77], q[81];
rzz(0.9097728133201599) q[78], q[81];
rzz(0.25301069021224976) q[79], q[81];
rzz(0.9193097949028015) q[80], q[81];
h q[0];