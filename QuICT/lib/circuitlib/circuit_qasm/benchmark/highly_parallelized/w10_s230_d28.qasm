OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rz(0.7735725411126332) q[4];
rz(3.1451055563893155) q[6];
rz(5.408065785102935) q[2];
rz(2.283205535265103) q[7];
cx q[8], q[9];
rz(4.055888729110017) q[1];
rz(5.421412597113007) q[5];
rz(4.21835487794747) q[0];
rz(2.8362121661954927) q[3];
cx q[2], q[5];
rz(1.7217568386402102) q[3];
cx q[0], q[8];
cx q[1], q[7];
rz(0.8367741145802762) q[6];
rz(0.4326504973485996) q[9];
rz(1.6213134078581966) q[4];
rz(4.5373554253166315) q[8];
rz(1.300924110203513) q[7];
cx q[5], q[9];
rz(4.892207540357602) q[1];
rz(0.21256856950322367) q[3];
rz(5.736304501544548) q[0];
rz(5.437361983400114) q[6];
cx q[4], q[2];
rz(4.749697411964553) q[1];
rz(4.035613819021533) q[4];
rz(0.6775937138242287) q[6];
rz(5.4711539993178855) q[5];
rz(2.1784800208273256) q[2];
rz(1.7650294613897421) q[8];
cx q[0], q[3];
rz(3.4852300554868405) q[7];
rz(3.84120083508575) q[9];
rz(1.1574700410309198) q[0];
rz(2.37840010820355) q[5];
cx q[3], q[4];
rz(3.59532732479519) q[9];
rz(4.213208342954967) q[8];
rz(3.2351958585057976) q[2];
rz(5.899351860934349) q[7];
rz(5.213686948632601) q[1];
rz(0.1082715087867377) q[6];
rz(1.2472581407541388) q[0];
rz(1.2937182563311613) q[5];
rz(2.1808046904697913) q[3];
rz(0.6277998321658722) q[7];
rz(4.725991804446855) q[1];
cx q[9], q[8];
rz(5.045826896737834) q[4];
rz(1.5196292506067288) q[6];
rz(0.8008240194264107) q[2];
rz(2.556917147433453) q[1];
rz(3.982168599689577) q[5];
rz(1.4511905308836057) q[3];
cx q[9], q[2];
rz(2.6154387080450947) q[8];
rz(3.370567769526263) q[0];
rz(0.46417606686290913) q[4];
rz(2.1971944817533156) q[6];
rz(6.043803418554466) q[7];
rz(1.2250259093621787) q[2];
rz(5.600275444810857) q[5];
rz(4.828053959887427) q[9];
cx q[7], q[0];
rz(5.8928039408000465) q[6];
rz(0.027251317088065573) q[8];
rz(0.31940759713601574) q[4];
rz(5.671038065175018) q[3];
rz(0.8456731111126996) q[1];
rz(0.09288090152806347) q[3];
rz(5.973531194664025) q[8];
rz(5.606924009607603) q[2];
rz(2.539292460415364) q[0];
rz(4.821007921514814) q[6];
rz(3.741970207282835) q[7];
rz(0.43215559465936393) q[4];
rz(2.1240086818441113) q[9];
rz(3.157272936678795) q[1];
rz(1.0829939507520787) q[5];
cx q[0], q[8];
rz(4.642637893536865) q[5];
cx q[3], q[7];
rz(4.864428678625777) q[2];
rz(1.2083509250284545) q[6];
rz(2.8297238170301613) q[9];
rz(2.2105085104494435) q[4];
rz(3.7400981154512745) q[1];
rz(4.327327209541768) q[9];
rz(5.965171555518113) q[4];
rz(0.2376105530518224) q[2];
rz(2.9624054713867145) q[3];
rz(5.796343483753623) q[6];
rz(2.7117766791067046) q[7];
cx q[8], q[0];
rz(0.38797417658923344) q[5];
rz(1.189238254431465) q[1];
rz(5.668082900083253) q[5];
cx q[8], q[4];
rz(0.191919610222911) q[7];
rz(0.2789916382557648) q[6];
cx q[1], q[0];
rz(1.8177844550632518) q[2];
cx q[3], q[9];
rz(2.2708307107908134) q[8];
rz(3.6995052327941704) q[9];
rz(4.681361041215051) q[3];
rz(1.360461698611869) q[4];
rz(4.575268816490889) q[2];
rz(0.2431350902743788) q[5];
rz(0.003819287129277446) q[0];
cx q[7], q[6];
rz(0.008998769590768618) q[1];
rz(4.163986452902758) q[4];
rz(4.046033905629494) q[2];
cx q[3], q[0];
cx q[1], q[6];
rz(5.316053649365318) q[9];
cx q[8], q[5];
rz(0.6695004305663391) q[7];
rz(0.5861004465615077) q[7];
cx q[9], q[8];
rz(4.1749837477112735) q[1];
rz(4.778095519772351) q[6];
cx q[0], q[3];
rz(6.23413281451774) q[4];
rz(1.1898757764655241) q[2];
rz(2.1532408544815693) q[5];
rz(1.403487551690425) q[4];
rz(2.4303831076692344) q[5];
rz(2.8871439378875055) q[2];
rz(2.551903792439656) q[3];
rz(3.3258103026666204) q[1];
rz(2.0747969140378144) q[9];
rz(3.29087117179167) q[0];
rz(0.29399384109522625) q[6];
cx q[7], q[8];
rz(5.581737946086474) q[3];
cx q[9], q[7];
cx q[8], q[1];
rz(3.4023084523359715) q[0];
rz(5.4215121770159875) q[4];
rz(6.227083495700642) q[6];
rz(0.035152938413275533) q[2];
rz(5.422609147751234) q[5];
cx q[4], q[0];
rz(5.691009351473376) q[7];
rz(5.49591211407371) q[3];
rz(1.2904561891959263) q[9];
rz(3.4105190798792075) q[2];
rz(1.698857773431984) q[6];
rz(3.9251347840165045) q[5];
rz(0.41332745180074454) q[1];
rz(3.7081528466161027) q[8];
rz(4.1848033398184095) q[3];
rz(2.1195734399535127) q[2];
rz(4.511336509465524) q[6];
cx q[1], q[7];
rz(4.059758737886064) q[5];
cx q[4], q[9];
rz(3.62838239457391) q[0];
rz(4.739491295164433) q[8];
rz(5.161739163981555) q[3];
cx q[2], q[5];
rz(1.9503112315548783) q[0];
cx q[9], q[7];
rz(0.9752203270186082) q[6];
rz(0.7499405508845194) q[4];
cx q[1], q[8];
rz(3.708581682815353) q[7];
rz(0.26119569505674234) q[2];
rz(1.2700292862516525) q[0];
rz(4.167284699567422) q[6];
cx q[3], q[5];
rz(1.39794880643259) q[1];
rz(3.15290613757635) q[4];
cx q[9], q[8];
rz(5.773777546580798) q[9];
rz(1.0143680267856034) q[1];
cx q[2], q[6];
cx q[5], q[7];
rz(2.148960781013659) q[4];
cx q[8], q[3];
rz(5.66217139303639) q[0];
cx q[8], q[4];
rz(2.705386503130936) q[2];
cx q[0], q[1];
rz(4.358440917283418) q[7];
rz(5.956760035995175) q[6];
rz(4.412577122237807) q[9];
rz(1.5580881289273079) q[5];
rz(2.9704787429039827) q[3];
cx q[8], q[4];
rz(4.8373363191526755) q[6];
rz(3.4662137134558697) q[0];
rz(6.110285866083001) q[1];
rz(2.518549927551972) q[2];
rz(1.0520241983260474) q[5];
rz(2.75893396094116) q[7];
rz(2.6184461269704475) q[3];
rz(4.935336398728447) q[9];
rz(2.5619479759133283) q[8];
rz(3.6354015909298236) q[7];
rz(3.4753048025459607) q[2];
rz(0.24156554027135185) q[3];
rz(1.6719289805751296) q[1];
rz(0.6251217471209484) q[6];
rz(6.0194155156602545) q[5];
rz(2.104002437815576) q[9];
rz(0.9701482296354574) q[0];
rz(5.953259518689054) q[4];
rz(3.0503185794259027) q[4];
cx q[8], q[1];
rz(0.8843863882308655) q[3];
cx q[6], q[7];
rz(3.6202079248031924) q[9];
cx q[2], q[5];
rz(2.500877995216765) q[0];
rz(5.960713699636698) q[6];
rz(4.903357032928174) q[8];
rz(0.7321397838272802) q[5];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];