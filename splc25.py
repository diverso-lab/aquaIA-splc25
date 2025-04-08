from flamapy.core.discover import DiscoverMetamodels


def xml_to_uvl():
    for file in 'models/betty/*':
        dm = DiscoverMetamodels()
        feature_model = dm.use_transformation_t2m(file,'fm')
        print(feature_model)
        dm.use_transformation_m2t(feature_model,'./models/uvl/'+file.split('/')[-1].split('.')[0]+'.uvl')
