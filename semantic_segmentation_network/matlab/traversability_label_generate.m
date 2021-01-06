clear;
clc;

root = "/media/youngji/StorageDevice/data/kitti_traversability_new/";
rgb_folder = root + "rgb/";
segmentation_folder = root + "seg/";
projection_folder = root + "proj/";

save_root = "/media/youngji/StorageDevice/data/nvidia-segmentation/data_semantics/training/";
label_train_folder = save_root + "semantic/";
image_train_folder = save_root + "image_2/";

% label_val_folder = root + "labels/val/";
% image_val_folder = root + "images/val/";

train_seq_names = ["kitti06","kitti07"];%["kitti06","kitti07","kitti09","kitti10"];
% val_seq_names = ["kitti06"];
% val_seq_names = ["kitti15"];


for iter = 1:numel(train_seq_names)
    images = dir(fullfile(projection_folder + train_seq_names(iter) + "/", '*.png'));

    % Generate training data
    for i = 1: size(images)
        image_name = images(i).name;
        rgb_img = imread(rgb_folder + train_seq_names(iter) + "/" + image_name);
        try
            segmentation_img = imread(segmentation_folder + train_seq_names(iter)+ "/" + image_name);
            projection_img = imread(projection_folder + train_seq_names(iter)+ "/" + image_name);
            final_img = uint8(and(segmentation_img, projection_img));
            final_img = final_img + 34;
            %resize 
            final_img = imresize(final_img,[375,1242],'nearest');
            rgb_img = imresize(rgb_img,[375,1242],'nearest');
            %save images
            imwrite(final_img, label_train_folder + train_seq_names(iter) + "_" + image_name);
            imwrite(rgb_img, image_train_folder + train_seq_names(iter) + "_" + image_name);
        catch
            continue
        end
    end
end

% %% Generate validation data
% percentage = 0.05;
% for iter=1:numel(val_seq_names)
%     images = dir(fullfile(projection_folder + val_seq_names(iter) + "/", '*.png'));
% %     valid_size = round(percentage * size(images, 1));
%     valid_images = 1:1:30;%round(1+rand(valid_size, 1) * valid_size);
%     for i = 1 : numel(valid_images)
%         image_name = images(valid_images(i)).name;
%         rgb_img = imread(rgb_folder + val_seq_names(iter) + "/" + image_name);
%         try
%             segmentation_img = imread(segmentation_folder + val_seq_names(iter) + "/" + image_name);
%             mask_img = imread(mask_folder + val_seq_names(iter) + "/" + image_name);
%             projection_img = imread(projection_folder + val_seq_names(iter) + "/" + image_name);
%             final_img = uint8(and(segmentation_img, projection_img));
%             weight_img = uint8(or(~segmentation_img, mask_img));
%             %resize
%             final_img = imresize(final_img,[480,640],'nearest');
%             rgb_img = imresize(rgb_img,[480,640],'nearest');
%             weight_img = imresize(weight_img,[480,640],'nearest');
%             %imagesc(final_img)
%             imwrite(final_img, label_val_folder + val_seq_names(iter) + "_" + image_name);
%             imwrite(rgb_img, image_val_folder + val_seq_names(iter) + "_" + image_name);
%             imwrite(weight_img, weight_val_folder + val_seq_names(iter) + "_" + image_name);
%         catch
%             continue
%         end
%     end
% end

% %% Generate test data
% % percentage = 0.05;
% for iter=1:numel(val_seq_names)
%     images = dir(fullfile(mask_folder + val_seq_names(iter) + "/", '*.png'));
%     for i = 1 : size(images)
%         image_name = images(i).name;
% %         rgb_img = imread(rgb_folder + val_seq_names(iter) + "/" + image_name);
%         try
%             segmentation_img = imread(segmentation_folder + val_seq_names(iter) + "/" + image_name);
%             mask_img = imread(mask_folder + val_seq_names(iter) + "/" + image_name);
%             projection_img = imread(projection_folder + val_seq_names(iter) + "/" + image_name);
%             final_img = uint8(and(segmentation_img, projection_img));
%             weight_img = uint8(or(~segmentation_img, mask_img));
%             %resize
% %             final_img = imresize(final_img,[480,640]);
% %             rgb_img = imresize(rgb_img,[480,640]);
% %             weight_img = imresize(weight_img,[480,640]);
%             %imagesc(final_img)
%             imwrite(final_img, root + "kitti15/" + image_name);
% %             imwrite(rgb_img, image_val_folder + val_seq_names(iter) + "_" + image_name);
% %             imwrite(weight_img, weight_val_folder + val_seq_names(iter) + "_" + image_name);
%         catch
%             continue
%         end
%     end
% end