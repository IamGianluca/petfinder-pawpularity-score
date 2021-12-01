from typing import List

import pandas as pd

import constants

# https://www.kaggle.com/valleyzw/petfinder-duplicate-images
similar_images = [
    ("13d215b4c71c3dc603cd13fc3ec80181", "373c763f5218610e9b3f82b12ada8ae5"),
    ("5ef7ba98fc97917aec56ded5d5c2b099", "67e97de8ec7ddcda59a58b027263cdcc"),
    ("839087a28fa67bf97cdcaf4c8db458ef", "a8f044478dba8040cc410e3ec7514da1"),
    ("1feb99c2a4cac3f3c4f8a4510421d6f5", "264845a4236bc9b95123dde3fb809a88"),
    ("3c50a7050df30197e47865d08762f041", "def7b2f2685468751f711cc63611e65b"),
    ("37ae1a5164cd9ab4007427b08ea2c5a3", "3f0222f5310e4184a60a7030da8dc84b"),
    ("5a642ecc14e9c57a05b8e010414011f2", "c504568822c53675a4f425c8e5800a36"),
    ("2a8409a5f82061e823d06e913dee591c", "86a71a412f662212fe8dcd40fdaee8e6"),
    ("3c602cbcb19db7a0998e1411082c487d", "a8bb509cd1bd09b27ff5343e3f36bf9e"),
    ("0422cd506773b78a6f19416c98952407", "0b04f9560a1f429b7c48e049bcaffcca"),
    ("68e55574e523cf1cdc17b60ce6cc2f60", "9b3267c1652691240d78b7b3d072baf3"),
    ("1059231cf2948216fcc2ac6afb4f8db8", "bca6811ee0a78bdcc41b659624608125"),
    ("5da97b511389a1b62ef7a55b0a19a532", "8ffde3ae7ab3726cff7ca28697687a42"),
    ("78a02b3cb6ed38b2772215c0c0a7f78e", "c25384f6d93ca6b802925da84dfa453e"),
    ("08440f8c2c040cf2941687de6dc5462f", "bf8501acaeeedc2a421bac3d9af58bb7"),
    ("0c4d454d8f09c90c655bd0e2af6eb2e5", "fe47539e989df047507eaa60a16bc3fd"),
    ("5a5c229e1340c0da7798b26edf86d180", "dd042410dc7f02e648162d7764b50900"),
    ("871bb3cbdf48bd3bfd5a6779e752613e", "988b31dd48a1bc867dbc9e14d21b05f6"),
    ("dbf25ce0b2a5d3cb43af95b2bd855718", "e359704524fa26d6a3dcd8bfeeaedd2e"),
    ("43bd09ca68b3bcdc2b0c549fd309d1ba", "6ae42b731c00756ddd291fa615c822a1"),
    ("43ab682adde9c14adb7c05435e5f2e0e", "9a0238499efb15551f06ad583a6fa951"),
    ("a9513f7f0c93e179b87c01be847b3e4c", "b86589c3e85f784a5278e377b726a4d4"),
    ("38426ba3cbf5484555f2b5e9504a6b03", "6cb18e0936faa730077732a25c3dfb94"),
    ("589286d5bfdc1b26ad0bf7d4b7f74816", "cd909abf8f425d7e646eebe4d3bf4769"),
    ("9f5a457ce7e22eecd0992f4ea17b6107", "b967656eb7e648a524ca4ffbbc172c06"),
    ("b148cbea87c3dcc65a05b15f78910715", "e09a818b7534422fb4c688f12566e38f"),
    ("3877f2981e502fe1812af38d4f511fd2", "902786862cbae94e890a090e5700298b"),
    ("8f20c67f8b1230d1488138e2adbb0e64", "b190f25b33bd52a8aae8fd81bd069888"),
    ("221b2b852e65fe407ad5fd2c8e9965ef", "94c823294d542af6e660423f0348bf31"),
    ("2b737750362ef6b31068c4a4194909ed", "41c85c2c974cc15ca77f5ababb652f84"),
    ("01430d6ae02e79774b651175edd40842", "6dc1ae625a3bfb50571efedc0afc297c"),
    ("72b33c9c368d86648b756143ab19baeb", "763d66b9cf01069602a968e573feb334"),
    ("03d82e64d1b4d99f457259f03ebe604d", "dbc47155644aeb3edd1bd39dba9b6953"),
    ("851c7427071afd2eaf38af0def360987", "b49ad3aac4296376d7520445a27726de"),
    ("54563ff51aa70ea8c6a9325c15f55399", "b956edfd0677dd6d95de6cb29a85db9c"),
    ("87c6a8f85af93b84594a36f8ffd5d6b8", "d050e78384bd8b20e7291b3efedf6a5b"),
    ("04201c5191c3b980ae307b20113c8853", "16d8e12207ede187e65ab45d7def117b"),
]


def flag_duplicates_and_impute_mean(duplicates: List[List], df: pd.DataFrame):
    """For every duplicate pair/triplet in the training dataset, keep one image
    and replace the target value to the average of the original images.
    """
    df["keep"] = 1
    print(f"There are {len(duplicates)} duplicate images")

    for pair in duplicates:
        pawpularity = df[df.Id.isin(pair)].Pawpularity.mean()
        df.loc[df.Id.isin(pair), "Pawpularity"] = int(pawpularity)
        df.loc[df.Id.isin(pair[1:]), "keep"] = 0

    df = df[df.keep == 1]
    df = df.drop("keep", axis=1)
    return df


if __name__ == "__main__":
    df = pd.read_csv(constants.train_labels_fpath)
    print(f"Original DataFrame shape: {df.shape}")

    df = flag_duplicates_and_impute_mean(duplicates=similar_images, df=df)

    print(f"New DataFrame shape: {df.shape}")
    df.to_csv(constants.train_deduped_fpath)
