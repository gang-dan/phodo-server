package app.gangdan.phodoapi.domain.review;

import lombok.*;

import javax.persistence.*;

@Entity
@Table(name = "review")
@Getter
@Builder
@NoArgsConstructor @AllArgsConstructor
public class Review {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long reviewId;

    private String reviewContent;

    private Double reviewScore;

}
