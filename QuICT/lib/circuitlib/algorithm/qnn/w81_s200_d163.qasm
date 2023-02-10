OPENQASM 2.0;
include "qelib1.inc";
qreg q[81];
creg c[81];
x q[0];
x q[1];
x q[5];
x q[6];
x q[12];
x q[13];
x q[15];
x q[19];
x q[21];
x q[22];
x q[23];
x q[25];
x q[26];
x q[27];
x q[28];
x q[31];
x q[33];
x q[34];
x q[35];
x q[36];
x q[37];
x q[39];
x q[40];
x q[41];
x q[43];
x q[47];
x q[52];
x q[56];
x q[57];
x q[59];
x q[62];
x q[66];
x q[68];
x q[76];
x q[77];
x q[78];
x q[79];
x q[0];
h q[0];
rzz(0.3805696964263916) q[0], q[80];
rzz(0.4271298050880432) q[1], q[80];
rzz(0.7668089270591736) q[2], q[80];
rzz(0.9711045026779175) q[3], q[80];
rzz(0.34300893545150757) q[4], q[80];
rzz(0.42087531089782715) q[5], q[80];
rzz(0.5087261199951172) q[6], q[80];
rzz(0.8845434784889221) q[7], q[80];
rzz(0.04680269956588745) q[8], q[80];
rzz(0.4121744632720947) q[9], q[80];
rzz(0.04430431127548218) q[10], q[80];
rzz(0.12307673692703247) q[11], q[80];
rzz(0.12656104564666748) q[12], q[80];
rzz(0.5998331308364868) q[13], q[80];
rzz(0.8770155310630798) q[14], q[80];
rzz(0.7350497245788574) q[15], q[80];
rzz(0.33356624841690063) q[16], q[80];
rzz(0.35455322265625) q[17], q[80];
rzz(0.7284473776817322) q[18], q[80];
rzz(0.910838782787323) q[19], q[80];
rzz(0.3837118148803711) q[20], q[80];
rzz(0.8772281408309937) q[21], q[80];
rzz(0.7208940386772156) q[22], q[80];
rzz(0.4835411310195923) q[23], q[80];
rzz(0.8634487986564636) q[24], q[80];
rzz(0.8836750984191895) q[25], q[80];
rzz(0.6027088761329651) q[26], q[80];
rzz(0.8552121520042419) q[27], q[80];
rzz(0.06063908338546753) q[28], q[80];
rzz(0.4902889132499695) q[29], q[80];
rzz(0.12339478731155396) q[30], q[80];
rzz(0.6967931389808655) q[31], q[80];
rzz(0.6128411293029785) q[32], q[80];
rzz(0.2648344039916992) q[33], q[80];
rzz(0.3564645051956177) q[34], q[80];
rzz(0.4776309132575989) q[35], q[80];
rzz(0.665608823299408) q[36], q[80];
rzz(0.6526661515235901) q[37], q[80];
rzz(0.6135146021842957) q[38], q[80];
rzz(0.7135574817657471) q[39], q[80];
rzz(0.9565401673316956) q[40], q[80];
rzz(0.11248654127120972) q[41], q[80];
rzz(0.13327056169509888) q[42], q[80];
rzz(0.5285362601280212) q[43], q[80];
rzz(0.07739901542663574) q[44], q[80];
rzz(0.44255876541137695) q[45], q[80];
rzz(0.6645210385322571) q[46], q[80];
rzz(0.12040120363235474) q[47], q[80];
rzz(0.8076812028884888) q[48], q[80];
rzz(0.4689399003982544) q[49], q[80];
rzz(0.6419083476066589) q[50], q[80];
rzz(0.6498162746429443) q[51], q[80];
rzz(0.9663465023040771) q[52], q[80];
rzz(0.9629557132720947) q[53], q[80];
rzz(0.9028809666633606) q[54], q[80];
rzz(0.7903597950935364) q[55], q[80];
rzz(0.8843705058097839) q[56], q[80];
rzz(0.2681514024734497) q[57], q[80];
rzz(0.8208468556404114) q[58], q[80];
rzz(0.33573490381240845) q[59], q[80];
rzz(0.8067183494567871) q[60], q[80];
rzz(0.5231428146362305) q[61], q[80];
rzz(0.6869139671325684) q[62], q[80];
rzz(0.8542764186859131) q[63], q[80];
rzz(0.19957178831100464) q[64], q[80];
rzz(0.632223904132843) q[65], q[80];
rzz(0.3234363794326782) q[66], q[80];
rzz(0.5968523621559143) q[67], q[80];
rzz(0.8256802558898926) q[68], q[80];
rzz(0.001259446144104004) q[69], q[80];
rzz(0.21211963891983032) q[70], q[80];
rzz(0.7287337779998779) q[71], q[80];
rzz(0.4925953149795532) q[72], q[80];
rzz(0.5053490996360779) q[73], q[80];
rzz(0.8886327743530273) q[74], q[80];
rzz(0.8198692798614502) q[75], q[80];
rzz(0.3138159513473511) q[76], q[80];
rzz(0.9180207848548889) q[77], q[80];
rzz(0.5506426095962524) q[78], q[80];
rzz(0.5094151496887207) q[79], q[80];
rzz(0.2164803147315979) q[0], q[80];
rzz(0.5559970736503601) q[1], q[80];
rzz(0.4731232523918152) q[2], q[80];
rzz(0.981059193611145) q[3], q[80];
rzz(0.48191118240356445) q[4], q[80];
rzz(0.8370347023010254) q[5], q[80];
rzz(0.671385645866394) q[6], q[80];
rzz(0.053009748458862305) q[7], q[80];
rzz(0.7386809587478638) q[8], q[80];
rzz(0.06089299917221069) q[9], q[80];
rzz(0.1887637972831726) q[10], q[80];
rzz(0.4661791920661926) q[11], q[80];
rzz(0.5091684460639954) q[12], q[80];
rzz(0.3342372179031372) q[13], q[80];
rzz(0.31464093923568726) q[14], q[80];
rzz(0.33868157863616943) q[15], q[80];
rzz(0.38530290126800537) q[16], q[80];
rzz(0.6420731544494629) q[17], q[80];
rzz(0.15946048498153687) q[18], q[80];
rzz(0.08545726537704468) q[19], q[80];
rzz(0.32226109504699707) q[20], q[80];
rzz(0.4038820266723633) q[21], q[80];
rzz(0.04446816444396973) q[22], q[80];
rzz(0.5389629006385803) q[23], q[80];
rzz(0.08351194858551025) q[24], q[80];
rzz(0.5307842493057251) q[25], q[80];
rzz(0.4659087061882019) q[26], q[80];
rzz(0.7518042325973511) q[27], q[80];
rzz(0.6710518002510071) q[28], q[80];
rzz(0.6214489340782166) q[29], q[80];
rzz(0.5021688342094421) q[30], q[80];
rzz(0.4343751072883606) q[31], q[80];
rzz(0.2493818998336792) q[32], q[80];
rzz(0.5481497645378113) q[33], q[80];
rzz(0.324465811252594) q[34], q[80];
rzz(0.2502697706222534) q[35], q[80];
rzz(0.4713020324707031) q[36], q[80];
rzz(0.28494250774383545) q[37], q[80];
rzz(0.031938791275024414) q[38], q[80];
rzz(0.5881990194320679) q[39], q[80];
rzz(0.15381526947021484) q[40], q[80];
rzz(0.3192253112792969) q[41], q[80];
rzz(0.7269022464752197) q[42], q[80];
rzz(0.019708633422851562) q[43], q[80];
rzz(0.45242881774902344) q[44], q[80];
rzz(0.2841116786003113) q[45], q[80];
rzz(0.029519617557525635) q[46], q[80];
rzz(0.5379951596260071) q[47], q[80];
rzz(0.9723926186561584) q[48], q[80];
rzz(0.39850765466690063) q[49], q[80];
rzz(0.4285036325454712) q[50], q[80];
rzz(0.7412006855010986) q[51], q[80];
rzz(0.18297016620635986) q[52], q[80];
rzz(0.9789952039718628) q[53], q[80];
rzz(0.3256537914276123) q[54], q[80];
rzz(0.2967001795768738) q[55], q[80];
rzz(0.5768464803695679) q[56], q[80];
rzz(0.6300784945487976) q[57], q[80];
rzz(0.9648069739341736) q[58], q[80];
rzz(0.8985700011253357) q[59], q[80];
rzz(0.5301797389984131) q[60], q[80];
rzz(0.550302267074585) q[61], q[80];
rzz(0.1607116460800171) q[62], q[80];
rzz(0.15486770868301392) q[63], q[80];
rzz(0.3972957134246826) q[64], q[80];
rzz(0.2782009243965149) q[65], q[80];
rzz(0.9171374440193176) q[66], q[80];
rzz(0.425937294960022) q[67], q[80];
rzz(0.500940203666687) q[68], q[80];
rzz(0.4775547385215759) q[69], q[80];
rzz(0.6159535050392151) q[70], q[80];
rzz(0.6013120412826538) q[71], q[80];
rzz(0.4816233515739441) q[72], q[80];
rzz(0.8777657747268677) q[73], q[80];
rzz(0.29359352588653564) q[74], q[80];
rzz(0.866089403629303) q[75], q[80];
rzz(0.5637972950935364) q[76], q[80];
rzz(0.3150608539581299) q[77], q[80];
rzz(0.8408268094062805) q[78], q[80];
rzz(0.23908275365829468) q[79], q[80];
h q[0];