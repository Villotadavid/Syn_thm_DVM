#include "dronet_control/deep_navigation.h"
#include <iostream>
#include <fstream>




using std::fstream;

namespace deep_navigation
{

deepNavigation::deepNavigation(
    const ros::NodeHandle& nh,
    const ros::NodeHandle& nh_private)
  :   nh_(nh),
      nh_private_(nh_private),
      name_(nh_private.getNamespace())
{
  ROS_INFO("[%s]: Initializing Deep Control Node", name_.c_str());
  loadParameters();


  deep_network_sub_ = nh_.subscribe("cnn_predictions", 1, &deepNavigation::deepNetworkCallback, this);
  Coll_sub_ = nh_.subscribe("/Colision", 1, &deepNavigation::CollisionCallback, this);
  state_change_sub_ = nh_.subscribe("state_change", 1, &deepNavigation::stateChangeCallback, this);
  //ros::Publisher velocity_params = nh_.advertise < geometry_msgs::Twist > ("/hola", 10);
  //ros::Publisher desired_velocity_pub_ = nh_.advertise < geometry_msgs::Twist > ("velocity", 5);

  X = 0.0;
  Y = 0.0;
  X1 = 0.0;
  Y1 = 0.0;
  X2 = 0.0;
  Y2 = 0.0;
  // Aggressive initialization
  desired_forward_velocity_ = max_forward_index_;
  desired_angular_velocity_ = 0.0;

  use_network_out_ = false;

}


void deepNavigation::run()
{

  ros::Duration(2.0).sleep();
  ros::Publisher velocity_params = nh_.advertise < geometry_msgs::Twist > ("/hola", 10);
  ros::Publisher desired_velocity_pub_ = nh_.advertise < geometry_msgs::Twist > ("velocity", 5);
  ros::Rate rate(30.0);

  while (ros::ok())
  {

    Coll_=0.25*Coll1+0.1*Coll2+0.1*(Coll-Coll1)+Coll;
    Vx=-21.43*Coll_+19.29; 	//m/s
    Vy=(0.1*(X2)+0.25*(X1)+(X)+((X)-(X1)));
    Vz=(0.1*(Y2)+0.25*(Y1)+(Y)+((Y)-(Y1)));
    // Publish desired
    if (use_network_out_)
    {
        desired_velocity_pub_.publish(cmd_velocity_);
    }
    else
	//velocity_params.publish(cmd_velocity_);
        ROS_INFO("NOT PUBLISHING VELOCITY");

    ROS_INFO("Vx: %.3f - Vy: %.3f - Vz: %.3f - Collision:%.3f", Vx, Vy,Vz ,Coll_);
    ROS_INFO("--------------------------------------------------");


    rate.sleep();
    ros::spinOnce();

  }

}

void deepNavigation::deepNetworkCallback(const dronet_perception::CNN_out::ConstPtr& msg)
{

  X2=X1;
  X1=X;
  Y2=Y1;
  Y1=Y;
  X = msg->collision_prob;
  Y = msg->steering_angle;

  // Output modulation
  //if (steering_angle_ < -1.0) { steering_angle_ = -1.0;}
  //if (steering_angle_ > 1.0) { steering_angle_ = 1.0;}

}

void deepNavigation::CollisionCallback(const std_msgs::Float32& msg)
{
    //change current state
    Coll2=Coll1;
    Coll1=Coll;
    Coll = msg.data;
    Coll=Coll-Coll1;   //Cierre de lazo
    
}

void deepNavigation::stateChangeCallback(const std_msgs::Bool& msg)
{
    //change current state
    use_network_out_ = msg.data;
}

void deepNavigation::loadParameters()
{

  ROS_INFO("[%s]: Reading parameters", name_.c_str()); 
  nh_private_.param<double>("alpha_velocity", alpha_velocity_, 0.3);
  nh_private_.param<double>("alpha_yaw", alpha_yaw_, 0.5);
  nh_private_.param<double>("max_forward_index", max_forward_index_, 0.2);
  nh_private_.param<double>("critical_prob", critical_prob_coll_, 0.7);

}

} // namespace deep_navigation

int main(int argc, char** argv)
{
  ros::init(argc, argv, "deep_navigation");
  deep_navigation::deepNavigation dn;



  dn.run();



  return 0;
}
