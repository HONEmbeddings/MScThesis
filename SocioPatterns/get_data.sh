!/bin/bash

mkdir "Contacts in a workplace"
pushd "Contacts in a workplace"
wget http://www.sociopatterns.org/wp-content/uploads/2016/06/tij_InVS.dat_.zip
unzip tij_InVS.dat_.zip
rm tij_InVS.dat_.zip
wget http://www.sociopatterns.org/wp-content/uploads/2016/06/metadata_InVS13.txt
popd

mkdir "Primary school temporal"
pushd "Primary school temporal"
wget http://www.sociopatterns.org/wp-content/uploads/2015/09/primaryschool.csv.gz
gunzip primaryschool.csv.gz
wget http://www.sociopatterns.org/wp-content/uploads/2015/09/metadata_primaryschool.txt
popd

mkdir "Hospital ward"
pushd "Hospital ward"
wget http://www.sociopatterns.org/wp-content/uploads/2013/09/detailed_list_of_contacts_Hospital.dat_.gz
gunzip detailed_list_of_contacts_Hospital.dat_.gz
mv detailed_list_of_contacts_Hospital.dat_ detailed_list_of_contacts_Hospital.dat
popd