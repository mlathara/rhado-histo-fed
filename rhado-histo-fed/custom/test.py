from slides_to_tiles import slides_to_tiles

num_classes, train_files, validation_files = slides_to_tiles(
    "/nfs/home/mlathara/test_slides/*svs",
    0,
    32,
    "/nfs/home/mlathara/test_slides/output_tiles",
    90,
    299,
    25,
    5,
    "/nfs/home/mlathara/test_slides/labels",
    0.5,
)

print("Number classes: " + str(num_classes))
print("Training files: " + str(train_files))
print("Validation files: " + str(validation_files))
