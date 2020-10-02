require 'fileutils'

train_data_path = "/Volumes/ハードディスク/data/train/data.txt"
test_data_path = "/Volumes/ハードディスク/data/test/data.txt"

FileUtils.touch(train_data_path) unless FileTest.exist?(train_data_path)
FileUtils.touch(test_data_path) unless FileTest.exist?(test_data_path)

test_sakurai_data_paths = Dir.glob("/Volumes/ハードディスク/data/test/sakurai/*.jpg")
test_kiyoe_data_paths = Dir.glob("/Volumes/ハードディスク/data/test/kiyoe/*.jpg")
test_hukuyama_data_paths = Dir.glob("/Volumes/ハードディスク/data/test/hukuyama/*.jpg")
train_sakurai_data_paths = Dir.glob("/Volumes/ハードディスク/data/train/sakurai/*.jpg")
train_kiyoe_data_paths = Dir.glob("/Volumes/ハードディスク/data/train/kiyoe/*.jpg")
train_hukuyama_data_paths = Dir.glob("/Volumes/ハードディスク/data/train/hukuyama/*.jpg")


File.open(test_data_path, "w") do |f|
	test_sakurai_data_paths.each { |path| f.puts("#{path} 0") }
	test_kiyoe_data_paths.each { |path| f.puts("#{path} 1")}
	test_hukuyama_data_paths.each { |path| f.puts("#{path} 2")}
end
File.open(train_data_path, "w") do |f|
	train_sakurai_data_paths.each { |path| f.puts("#{path} 0")}
	train_kiyoe_data_paths.each { |path| f.puts("#{path} 1")}
	train_hukuyama_data_paths.each{ |path| f.puts("#{path} 2")}
end
